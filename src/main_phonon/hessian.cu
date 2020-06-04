/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/


/*----------------------------------------------------------------------------80
Use finite difference to calculate the hessian (force constants).
    H_ij^ab = [F_i^a(-) - F_i^a(+)] / [u_j^b(+) - u_j^b(-)]
Then calculate the dynamical matrices with different k points.
------------------------------------------------------------------------------*/


#include "hessian.cuh"
#include "cusolver_wrapper.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <vector>


void Hessian::compute
(char* input_dir, Atom* atom, Force* force)
{
    initialize(input_dir, atom->N);
    find_H(atom, force);

    if (num_kpoints == 1) // currently for Alex's GKMA calculations
    {
        find_D(atom);
        find_eigenvectors(input_dir);
    }
    else
    {
        find_dispersion(input_dir, atom->box, atom->cpu_position_per_atom);
    }
}


void Hessian::read_basis(char* input_dir, size_t N)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/basis.in");
    FILE *fid = fopen(file, "r");
    size_t count;
    count = fscanf(fid, "%zu", &num_basis);
    PRINT_SCANF_ERROR(count, 1, "Reading error for basis.in.");

    basis.resize(num_basis);
    mass.resize(num_basis);
    for (size_t m = 0; m < num_basis; ++m)
    {
        count = fscanf(fid, "%zu%lf", &basis[m], &mass[m]);
        PRINT_SCANF_ERROR(count, 2, "Reading error for basis.in.");
    }
    label.resize(N);
    for (size_t n = 0; n < N; ++n)
    {
        count = fscanf(fid, "%zu", &label[n]);
        PRINT_SCANF_ERROR(count, 1, "Reading error for basis.in.");
    }
    fclose(fid);
}


void Hessian::read_kpoints(char* input_dir)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/kpoints.in");
    FILE *fid = fopen(file, "r");
    size_t count;
    count = fscanf(fid, "%zu", &num_kpoints);
    PRINT_SCANF_ERROR(count, 1, "Reading error for kpoints.in.");

    kpoints.resize(num_kpoints * 3);
    for (size_t m = 0; m < num_kpoints; ++m)
    {
        count = fscanf(fid, "%lf%lf%lf", &kpoints[m * 3 + 0],
            &kpoints[m * 3 + 1], &kpoints[m * 3 + 2]);
        PRINT_SCANF_ERROR(count, 3, "Reading error for kpoints.in.");
    }
    fclose(fid);
}


void Hessian::initialize(char* input_dir, size_t N)
{
    read_basis(input_dir, N);
    read_kpoints(input_dir);
    size_t num_H = num_basis * N * 9;
    size_t num_D = num_basis * num_basis * 9 * num_kpoints;
    H.resize(num_H, 0.0);
    DR.resize(num_D, 0.0);
    if (num_kpoints > 1) // for dispersion calculation
    {
        DI.resize(num_D, 0.0);
    }
}


bool Hessian::is_too_far
(
    const Box& box,
    const std::vector<double>& cpu_position_per_atom,
    const size_t n1,
    const size_t n2
)
{
    const int number_of_atoms = cpu_position_per_atom.size() / 3;
    double x12 = cpu_position_per_atom[n2]
               - cpu_position_per_atom[n1];
    double y12 = cpu_position_per_atom[n2 + number_of_atoms]
               - cpu_position_per_atom[n1 + number_of_atoms];
    double z12 = cpu_position_per_atom[n2 + number_of_atoms * 2]
               - cpu_position_per_atom[n1 + number_of_atoms * 2];
    dev_apply_mic(box, x12, y12, z12);
    double d12_square = x12 * x12 + y12 * y12 + z12 * z12;
    return (d12_square > (cutoff * cutoff));
}


void Hessian::find_H(Atom* atom, Force* force)
{
    size_t N = atom->N;
    for (size_t nb = 0; nb < num_basis; ++nb)
    {
        size_t n1 = basis[nb];
        for (size_t n2 = 0; n2 < N; ++n2)
        {
            if(is_too_far(atom->box, atom->cpu_position_per_atom, n1, n2))
            {
                continue;
            }
            size_t offset = (nb * N + n2) * 9;
            find_H12(n1, n2, atom, force, H.data() + offset);
        }
    }
}


static void find_exp_ikr
(
    const size_t n1,
    const size_t n2,
    const double* k,
    const Box& box,
    const std::vector<double>& cpu_position_per_atom,
    double& cos_kr,
    double& sin_kr
)
{
    const int number_of_atoms = cpu_position_per_atom.size() / 3;
    double x12 = cpu_position_per_atom[n2]
               - cpu_position_per_atom[n1];
    double y12 = cpu_position_per_atom[n2 + number_of_atoms]
               - cpu_position_per_atom[n1 + number_of_atoms];
    double z12 = cpu_position_per_atom[n2 + number_of_atoms * 2]
               - cpu_position_per_atom[n1 + number_of_atoms * 2];
    dev_apply_mic(box, x12, y12, z12);
    double kr = k[0] * x12 + k[1] * y12 + k[2] * z12;
    cos_kr = cos(kr);
    sin_kr = sin(kr);
}


void Hessian::output_D(char* input_dir)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/D.out");
    FILE *fid = fopen(file, "w");
    for (size_t nk = 0; nk < num_kpoints; ++nk)
    {
        size_t offset = nk * num_basis * num_basis * 9;
        for (size_t n1 = 0; n1 < num_basis * 3; ++n1)
        {
            for (size_t n2 = 0; n2 < num_basis * 3; ++n2)
            {
                // cuSOLVER requires column-major
                fprintf(fid, "%g ", DR[offset + n1 + n2 * num_basis * 3]);
            }
            if (num_kpoints > 1)
            {
                for (size_t n2 = 0; n2 < num_basis * 3; ++n2)
                {
                    // cuSOLVER requires column-major
                    fprintf(fid, "%g ", DI[offset + n1 + n2 * num_basis * 3]);
                }
            }
            fprintf(fid, "\n");
        }
    }
    fclose(fid);
}


void Hessian::find_omega(FILE* fid, size_t offset)
{
    size_t dim = num_basis * 3;
    std::vector<double> W(dim);
    eig_hermitian_QR(dim, DR.data() + offset, DI.data() + offset, W.data());
    double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION*TIME_UNIT_CONVERSION);
    for (size_t n = 0; n < dim; ++n)
    {
        fprintf(fid, "%g ", W[n] * natural_to_THz);
    }
    fprintf(fid, "\n");
}


void Hessian::find_omega_batch(FILE* fid)
{
    size_t dim = num_basis * 3;
    std::vector<double> W(dim * num_kpoints);
    eig_hermitian_Jacobi_batch(dim, num_kpoints, DR.data(), DI.data(), W.data());
    double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION*TIME_UNIT_CONVERSION);
    for (size_t nk = 0; nk < num_kpoints; ++nk)
    {
        size_t offset = nk * dim;
        for (size_t n = 0; n < dim; ++n)
        {
            fprintf(fid, "%g ", W[offset + n] * natural_to_THz);
        }
        fprintf(fid, "\n");
    }
}


void Hessian::find_dispersion
(
    char* input_dir,
    const Box& box,
    const std::vector<double>& cpu_position_per_atom
)
{
    const int number_of_atoms = cpu_position_per_atom.size() / 3;

    char file_omega2[200];
    strcpy(file_omega2, input_dir);
    strcat(file_omega2, "/omega2.out");
    FILE *fid_omega2 = fopen(file_omega2, "w");
    for (size_t nk = 0; nk < num_kpoints; ++nk)
    {
        size_t offset = nk * num_basis * num_basis * 9;
        for (size_t nb = 0; nb < num_basis; ++nb)
        {
            size_t n1 = basis[nb];
            size_t label_1 = label[n1];
            double mass_1 = mass[label_1];
            for (size_t n2 = 0; n2 < number_of_atoms; ++n2)
            {
                if(is_too_far(box, cpu_position_per_atom, n1, n2)) continue;
                double cos_kr, sin_kr;
                find_exp_ikr
                (
                    n1,
                    n2,
                    kpoints.data() + nk * 3,
                    box,
                    cpu_position_per_atom,
                    cos_kr,
                    sin_kr
                );

                size_t label_2 = label[n2];
                double mass_2 = mass[label_2];
                double mass_factor = 1.0 / sqrt(mass_1 * mass_2);
                double* H12 = H.data() + (nb * number_of_atoms + n2) * 9;
                for (size_t a = 0; a < 3; ++a)
                {
                    for (size_t b = 0; b < 3; ++b)
                    {
                        size_t a3b = a * 3 + b;
                        size_t row = label_1 * 3 + a;
                        size_t col = label_2 * 3 + b;
                        // cuSOLVER requires column-major
                        size_t index = offset + col * num_basis * 3 + row;
                        DR[index] += H12[a3b] * cos_kr * mass_factor;
                        DI[index] += H12[a3b] * sin_kr * mass_factor;
                    }
                }
            }
        }
        if (num_basis > 10) { find_omega(fid_omega2, offset); } // > 32x32
    }
    output_D(input_dir);
    if (num_basis <= 10) { find_omega_batch(fid_omega2); } // <= 32x32
    fclose(fid_omega2);
}


void Hessian::find_H12
(size_t n1, size_t n2, Atom *atom, Force *force, double* H12)
{
    double dx2 = displacement * 2;
    double f_positive[3];
    double f_negative[3];
    for (size_t beta = 0; beta < 3; ++beta)
    {
        get_f
        (
            -displacement,
            n1,
            n2,
            beta,
            atom->box,
            atom->position_per_atom,
            atom->type,
            atom->group,
            atom->neighbor,
            atom->potential_per_atom,
            atom->force_per_atom,
            atom->virial_per_atom,
            force,
            f_negative
        );

        get_f
        (
            displacement,
            n1,
            n2,
            beta,
            atom->box,
            atom->position_per_atom,
            atom->type,
            atom->group,
            atom->neighbor,
            atom->potential_per_atom,
            atom->force_per_atom,
            atom->virial_per_atom,
            force,
            f_positive
        );


        for (size_t alpha = 0; alpha < 3; ++alpha)
        {
            size_t index = alpha * 3 + beta;
            H12[index] = (f_negative[alpha] - f_positive[alpha]) / dx2;
        }
    }
}


void Hessian::find_D(Atom* atom)
{
    for (size_t nb = 0; nb < num_basis; ++nb)
    {
        size_t n1 = basis[nb];
        size_t label_1 = label[n1];
        double mass_1 = mass[label_1];
        for (size_t n2 = 0; n2 < atom->N; ++n2)
        {
            if(is_too_far(atom->box, atom->cpu_position_per_atom, n1, n2))
            {
                continue;
            }

            size_t label_2 = label[n2];
            double mass_2 = mass[label_2];
            double mass_factor = 1.0 / sqrt(mass_1 * mass_2);
            double* H12 = H.data() + (nb * atom->N + n2) * 9;
            for (size_t a = 0; a < 3; ++a)
            {
                for (size_t b = 0; b < 3; ++b)
                {
                    size_t a3b = a * 3 + b;
                    size_t row = label_1 * 3 + a;
                    size_t col = label_2 * 3 + b;
                    // cuSOLVER requires column-major
                    size_t index = col * num_basis * 3 + row;
                    DR[index] += H12[a3b] * mass_factor;
                }
            }
        }
    }
}


void Hessian::find_eigenvectors(char* input_dir)
{
    char file_eigenvectors[200];
    strcpy(file_eigenvectors, input_dir);
    strcat(file_eigenvectors, "/eigenvector.out");
    FILE *fid_eigenvectors = my_fopen(file_eigenvectors, "w");

    size_t dim = num_basis * 3;
    std::vector<double> W(dim);
    std::vector<double> eigenvectors(dim * dim);
    eigenvectors_symmetric_Jacobi(dim, DR.data(), W.data(), eigenvectors.data());

    double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION*TIME_UNIT_CONVERSION);

    // output eigenvalues
    for(size_t n = 0; n < dim; n++)
    {
        fprintf(fid_eigenvectors, "%g ",  W[n] * natural_to_THz);
    }
    fprintf(fid_eigenvectors, "\n");

    // output eigenvectors
    for(size_t col = 0; col < dim; col++)
    {
        for (size_t a = 0; a < 3; a++)
        {
            for(size_t b = 0; b < num_basis; b++)
            {
                 size_t row = a + b * 3;
                 // column-major order from cuSolver
                 fprintf(fid_eigenvectors, "%g ",  eigenvectors[row+col*dim]);
            }
        }
        fprintf(fid_eigenvectors, "\n");
    }
    fclose(fid_eigenvectors);
}


void Hessian::get_f
(
    const double dx,
    const size_t n1,
    const size_t n2,
    const size_t beta,
    const Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    Neighbor& neighbor,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    Force *force,
    double* f
)
{
    const int number_of_atoms = type.size();

    shift_atom(dx, n2, beta, position_per_atom);

    force->compute
    (
        box,
        position_per_atom,
        type,
        group,
        neighbor,
        potential_per_atom,
        force_per_atom,
        virial_per_atom
    );

    size_t M = sizeof(double);
    CHECK(cudaMemcpy(f + 0, force_per_atom.data() + n1, M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(f + 1, force_per_atom.data() + n1 + number_of_atoms, M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(f + 2, force_per_atom.data() + n1 + number_of_atoms * 2, M, cudaMemcpyDeviceToHost));

    shift_atom(-dx, n2, beta, position_per_atom);
}


static __global__ void gpu_shift_atom(const double dx, double *x)
{
    x[0] += dx;
}


void Hessian::shift_atom
(
    const double dx,
    const size_t n2,
    const size_t beta,
    GPU_Vector<double>& position_per_atom
)
{
    const int number_of_atoms = position_per_atom.size() / 3;

    if (beta == 0)
    {
        gpu_shift_atom<<<1, 1>>>(dx, position_per_atom.data() + n2);
        CUDA_CHECK_KERNEL
    }
    else if (beta == 1)
    {
        gpu_shift_atom<<<1, 1>>>
        (
            dx,
            position_per_atom.data() + number_of_atoms + n2
        );
        CUDA_CHECK_KERNEL
    }
    else
    {
        gpu_shift_atom<<<1, 1>>>
        (
            dx,
            position_per_atom.data() + number_of_atoms * 2 + n2
        );
        CUDA_CHECK_KERNEL
    }
}


void Hessian::parse_cutoff(char **param, size_t num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("cutoff should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &cutoff))
    {
        PRINT_INPUT_ERROR("cutoff for hessian should be a number.\n");
    }
    if (cutoff <= 0)
    {
        PRINT_INPUT_ERROR("cutoff for hessian should be positive.\n");
    }
    printf("Cutoff distance for hessian = %g A.\n", cutoff);
}


void Hessian::parse_delta(char **param, size_t num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("delta should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &displacement))
    {
        PRINT_INPUT_ERROR("delta for hessian should be a number.\n");
    }
    if (displacement <= 0)
    {
        PRINT_INPUT_ERROR("delta for hessian should be positive.\n");
    }
    printf("delta for hessian = %g A.\n", displacement);
}


