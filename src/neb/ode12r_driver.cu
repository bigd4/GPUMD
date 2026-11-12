#include "neb.cuh"
#include "ode12r.h"
#include <limits>

void NEB::parse_ode12r()
{
  if (tangent_method_name != "modified" || dynamic_relaxation || climb ||
      find_min || need_relax || ina_local_relax_steps || ina_k ||
      (variable_cell && !remove_rotation) || (variable_cell && pressure.size() != 1))
    PRINT_INPUT_ERROR("ode12r requires modified tangent, fixed/rotation-free cell, isotropic pressure, and no climb, dyneb or endpoint/local relaxation.");
  if (energy_based_spacing && energy_spacing_damping!=1.0)
    PRINT_INPUT_ERROR("ode12r requires energy_spacing_damping 1 for rollback-safe forces.");
  for (int i=0;i<int(optimizer_opt.size());i+=2) {
    if (i+1>=int(optimizer_opt.size())) PRINT_INPUT_ERROR("Every ode12r option requires a value.");
    const std::string key(optimizer_opt[i]), arg(optimizer_opt[i+1]);
    if (key=="tangent_window" || key=="residual_extrapolation") {
      if (arg!="on" && arg!="off") PRINT_INPUT_ERROR("ODE switch requires on/off.");
      const bool enabled=arg=="on";
      if (key=="tangent_window") tangent_window=enabled;
      else ode_residual_extrapolation=enabled;
    } else if (key=="window_restart") {
      window_restart=arg;
    } else if (key=="window_hold") {
      if (!is_valid_int(optimizer_opt[i+1],&window_hold) || window_hold<1)
        PRINT_INPUT_ERROR("window_hold requires a positive number of accepted updates.");
    } else {
      double value;
      if (!is_valid_real(optimizer_opt[i+1],&value) || !std::isfinite(value) || value<=0)
        PRINT_INPUT_ERROR("ODE numeric options must be positive and finite.");
      if (key=="window_off") window_off=value;
      else if (key=="window_on") window_on=value;
      else if (key=="rtol") ode_rtol=value;
      else if (key=="h0") ode_h0=value;
      else if (key=="hmin") ode_hmin=value;
      else if (key=="max_move") ode_max_move=value;
      else if (key=="secant_guard" && value<=1) ode_secant_guard=value;
      else PRINT_INPUT_ERROR("Unknown/invalid ode12r option.");
    }
  }
  if (!(window_off<window_on) || !(ode_hmin<ode_h0))
    PRINT_INPUT_ERROR("Require window_off < window_on and hmin < h0.");
  if (!tangent_window && !window_restart.empty())
    PRINT_INPUT_ERROR("window_restart requires tangent_window on.");
}

void NEB::run_ode12r()
{
  ode_window_audit=true;
  minimizer_cell_metric_scale=cell_metric_scale_default;
  initialize_compute();
  int size=positions.size(); const int block=3*natoms_per_image, real=3*n_realatoms;
  std::vector<double> x(size), f(size), trial(size), ft(size);
  auto metric_scale = [&](int j) { return variable_cell && j%block >= real ? minimizer_cell_metric_scale : 1.0; };
  auto residual = [&](const std::vector<double>& v) {
    double r=0.0;
    for (int j=0;j<size;++j) {
      if (!std::isfinite(v[j])) PRINT_INPUT_ERROR("Non-finite ODE12r driving force.");
      r=std::max(r,std::abs(v[j])/metric_scale(j));
    }
    return r;
  };
  NEBTangentWindow::Gate gate(nimages-2,window_off,window_on);
  ode_gate=&gate;
  std::vector<double> unfiltered(size);
  std::vector<double> window_T(nimages-2),window_P(nimages-2),window_F(nimages-2);
  FILE* wlog=fopen("neb_window.out","w");
  if(!wlog) PRINT_INPUT_ERROR("Cannot open window log.");
  if(wlog) fprintf(wlog,"# evaluation image event active_before active_after force_latch geometry_latch raw_T physical_perp full drive distance_left distance_right bend_cos switches image_id\n");
  auto collect_window=[&](const std::vector<double>& full) {
    if(!wlog)return;
    for(int i=0;i<nimages-2;++i) {
      window_T[i]=window_P[i]=window_F[i]=0;
      for(int j=0;j<block;++j) {int k=i*block+j; double m=metric_scale(k);
        window_T[i]=std::max(window_T[i],std::abs(window_parallel[k])/m);
        window_P[i]=std::max(window_P[i],std::abs(window_perpendicular[k])/m);
        window_F[i]=std::max(window_F[i],std::abs(full[k])/m);
      }
    }
  };
  auto filter_window=[&](std::vector<double>& force) {
    if(wlog && tangent_window) for(int i=0;i<nimages-2;++i) if(!gate.active(i)) for(int j=0;j<block;++j) force[i*block+j]-=window_parallel[i*block+j];
  };
  auto record_window=[&](int event,const std::vector<int>& before,const std::vector<double>& drive) {
    if(!wlog)return;
    for(int i=0;i<nimages-2;++i) {
      double mx=0;for(int j=0;j<block;++j)mx=std::max(mx,std::abs(drive[i*block+j])/metric_scale(i*block+j));
      fprintf(wlog,"%d %d %d %d %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %d\n",step,i+1,event,before[i],gate.active(i),gate.force[i],gate.geometry[i],window_T[i],window_P[i],window_F[i],mx,window_distances[i],window_distances[i+1],window_bend[i],gate.switches[i],gate.ids[i]);
    }
  };
  auto update_gate=[&]() {
    if(tangent_window)return gate.update(window_T);
    gate.force.assign(nimages-2,1);return false;
  };
  auto resize_storage=[&]() {
    size=positions.size();x.resize(size);f.resize(size);trial.resize(size);ft.resize(size);unfiltered.resize(size);
    window_T.resize(nimages-2);window_P.resize(nimages-2);window_F.resize(nimages-2);
    if(int(gate.force.size())!=nimages-2)PRINT_INPUT_ERROR("INA gate/image mapping mismatch.");
  };
  FILE* top=fopen("neb_window_topology.out","w");if(!top)PRINT_INPUT_ERROR("Cannot create topology log.");
  fprintf(top,"# evaluation action index id force_latch geometry_latch switches total_images\n");fclose(top);
  FILE* out=fopen("neb_ode12r.out","w");
  if (!out) PRINT_INPUT_ERROR("Cannot open ODE12r diagnostics.");
  fprintf(out,"# evaluation accepted_updates event h error trial_residual accepted_residual PES_calls_including_endpoints max_move h_error h_ls fy yy ff raw_h_ls cooldown stall_triggers\n");
  printf("ODE12r residual_extrapolation=%s\n", ode_residual_extrapolation ? "on" : "off");
  printf("ODE12r: rtol=%g h0=%g hmin=%g max_move=%g cell_metric_scale=%g; budget includes initial, rejected and final verification evaluations.\n",ode_rtol,ode_h0,ode_hmin,ode_max_move,minimizer_cell_metric_scale);
  printf("ODE12r secant_guard=%g\n",ode_secant_guard);
  NEBODE12r::StagnationMonitor stagnation;
  // event: 0 rejected trial, 1 accepted trial, 2 initial/final fresh evaluation.
  compute(); positions.copy_to_host(x.data()); forces.copy_to_host(f.data());
  if(wlog) {collect_window(f); update_gate();
    if (!window_restart.empty()) {
      try { gate.load(window_restart, n_realatoms); }
      catch (const std::exception& e) { PRINT_INPUT_ERROR(e.what()); }
      update_gate();
    }
    auto before=gate.mask();filter_window(f);record_window(2,before,f);
  }
  double r=residual(f); print_info(r);
  fprintf(out,"0 0 2 0 0 %.17g %.17g %d 0 0 0 0 0 0 0 0 0\n",r,r,n_force_calc+2);
  step=1;
  int accepted=0,rejected=0,ina_rebuilds=0,qualified=0;
  auto maximum=[](const std::vector<double>& v) { return *std::max_element(v.begin(),v.end()); };
  auto window_satisfied=[&]() { return maximum(window_P)<=force_tolerance &&
    maximum(window_T)<=window_on && gate.inactive(); };
  printf("Tangential window: %s; off=%g on=%g; transverse tolerance=%g; hold=%d accepted updates.\n",
    tangent_window?"on":"off",window_off,window_on,force_tolerance,window_hold);
  double h=ode_h0>0.0 ? ode_h0 : std::max(ode_hmin,0.5*std::sqrt(ode_rtol)/std::max(r,1e-30));
  const char* reason="evaluation_budget";
  // Reserve one PES evaluation for a restored and freshly evaluated final path.
  while (step<max_steps-1 && !(tangent_window ? qualified>=window_hold : r<=force_tolerance)) {
    h=std::min(h,ode_max_move/std::max(r,1e-30));
    if (!(h>ode_hmin)) { reason="step_too_small"; break; }
    for (int j=0;j<size;++j) trial[j]=x[j]+h*f[j];
    positions.copy_from_host(trial.data());
    compute(); forces.copy_to_host(ft.data());
    auto window_before=gate.mask();
    if(wlog) {unfiltered=ft;collect_window(ft);filter_window(ft);}
    const double rt=residual(ft);
    double error=0.0,fy=0.0,yy=0.0,ff=0.0;
    for (int j=0;j<size;++j) {
      const double fj=f[j]/metric_scale(j), y=(f[j]-ft[j])/metric_scale(j);
      error=std::max(error,0.5*h*std::abs(y)); fy+=fj*y; yy+=y*y; ff+=fj*fj;
    }
    const auto decision=NEBODE12r::decide(h,r,rt,error,fy,yy,ode_rtol,ode_hmin,ode_residual_extrapolation && stagnation.cooldown==0,ode_secant_guard);
    const double displacement=h*r;
    const double old_r=r;
    const double raw_h_ls=h*fy/(yy+1e-10);
    if (ode_secant_guard>0.0) stagnation.observe(decision.accept,displacement,old_r,rt,force_tolerance);
    bool window_changed=false;
    if (decision.accept) {
      if(wlog) {window_changed=update_gate();ft=unfiltered;filter_window(ft);}
      record_window(1,window_before,ft);
      x.swap(trial); f.swap(ft); r=residual(f); ++accepted;
    }
    else {++rejected;record_window(0,window_before,ft);}
    fprintf(out,"%d %d %d %.17g %.17g %.17g %.17g %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %d\n",step,accepted,decision.accept?1:0,h,error,rt,r,n_force_calc+2,displacement,decision.h_error,decision.h_ls,fy,yy,ff,raw_h_ls,stagnation.cooldown,stagnation.triggers);
    // Only accepted, currently evaluated coordinates may enter a path snapshot.
    if (decision.accept) {
      print_info(r);
      if (step%dump_interval==0) write_neb_traj("dump_traj.xyz","a");
      if (step%peek_interval==0) { write_neb_traj("peek_traj.xyz","w"); write_energies(); }
    }
    h=window_changed ? std::min(decision.next_h,ode_h0>0 ? ode_h0 : .02) : decision.next_h; ++step;
    const int previous_generation=gate.generation;
    if(decision.accept && image_number_adjustment && step<max_steps-1) {
      // INA runs only on an accepted, freshly evaluated path. Its force stages
      // use the ORIGINAL full NEB residual in both control and window branches.
      fmax=residual(unfiltered);
      const int generation=gate.generation;
      adjust_image_number();
      if(gate.generation!=generation) {
        for(auto& img:images)img->set_calc(*p_force);
        initialize_compute();resize_storage();
        compute();
        positions.copy_to_host(x.data());forces.copy_to_host(f.data());collect_window(f);
        auto before=gate.mask();update_gate();filter_window(f);r=residual(f);record_window(3,before,f);
        fprintf(out,"%d %d 3 0 0 %.17g %.17g %d 0 0 0 0 0 0 0 0 0\n",step,accepted,r,r,n_force_calc+2);
        print_info(r);char filename[96];snprintf(filename,sizeof(filename),"ina_traj_%06d.xyz",step);write_neb_traj(filename,"w");
        h=ode_h0>0 ? ode_h0 : .02;stagnation=NEBODE12r::StagnationMonitor{};++ina_rebuilds;++step;
      }
    }
    if (tangent_window) {
      if (gate.generation!=previous_generation) qualified=0;
      else if (decision.accept) qualified=window_satisfied() ? qualified+1 : 0;
    }
    if (step%100==0) { fflush(out); fflush(stdout); }
  }
  const double cached_residual=r;
  positions.copy_from_host(x.data());
  compute(); forces.copy_to_host(ft.data());
  if(wlog) {collect_window(ft);auto before=gate.mask();filter_window(ft);record_window(2,before,ft);}
  r=residual(ft);
  double force_error=0.0;
  for (int j=0;j<size;++j) force_error=std::max(force_error,std::abs(ft[j]-f[j])/metric_scale(j));
  if (force_error>1e-8*(1.0+r)) PRINT_INPUT_ERROR("ODE12r accepted-force cache differs after rollback verification.");
  print_info(r);
  fprintf(out,"%d %d 2 0 0 %.17g %.17g %d 0 0 0 0 0 0 0 0 0\n",step,accepted,r,r,n_force_calc+2);
  ++step; fclose(out);
  if(wlog) fclose(wlog);
  if (tangent_window) {
    try { gate.save("window_state.txt", n_realatoms); }
    catch (const std::exception& e) { PRINT_INPUT_ERROR(e.what()); }
  }
  ode_gate=nullptr;
  FILE* ids=fopen("window_ids.txt","w");if(!ids)PRINT_INPUT_ERROR("Cannot save window IDs.");for(int id:gate.ids)fprintf(ids,"%d\n",id);fclose(ids);
  if (tangent_window && qualified>=window_hold && window_satisfied()) reason="window_converged";
  else if (!tangent_window && r<=force_tolerance) reason="full_force_converged";
  write_energies(); write_neb_traj("final_traj.xyz","w");
  FILE* summary=fopen("neb_ode12r_summary.out","w");
  if (!summary) PRINT_INPUT_ERROR("Cannot open ODE12r summary.");
  fprintf(summary,"evaluations %d\naccepted_updates %d\nrejected_trials %d\nPES_calls_including_endpoints %d\nfinal_residual %.17g\ncached_residual %.17g\nrollback_force_error %.17g\nstop_reason %s\nstall_triggers %d\n",step,accepted,rejected,n_force_calc+2,r,cached_residual,force_error,reason,stagnation.triggers);
  fprintf(summary,"ina_rebuilds %d\nfinal_images %d\n",ina_rebuilds,nimages);
  fprintf(summary,"physical_perp_residual %.17g\nraw_tangent_residual %.17g\nraw_full_residual %.17g\nwindow_qualified_updates %d\nwindow_hold %d\nwindow_enabled %d\n",
    maximum(window_P),maximum(window_T),maximum(window_F),qualified,window_hold,int(tangent_window));
  fclose(summary);
  printf("ODE12r stop: %s; physical transverse=%g, raw tangent=%g, full NEB=%g, drive=%g eV/A.\n",
    reason,maximum(window_P),maximum(window_T),maximum(window_F),r);
  if (tangent_window) printf("Window criterion does not imply full NEB force convergence.\n");
  ode_window_audit=false;
  printf("ODE12r finished: evaluations=%d accepted=%d rejected=%d fmax=%.10g reason=%s.\n",step,accepted,rejected,r,reason);
}
