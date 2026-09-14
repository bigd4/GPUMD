#include <binary_xyz/binary_xyz.hpp>
#include <openssl/sha.h>
#include <zstd.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <utility>

namespace binary_xyz {
namespace {
using Bytes = std::vector<std::uint8_t>;
static_assert(sizeof(double) == 8 && sizeof(float) == 4 && std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<float>::is_iec559, "IEEE 754 floating point required");
const std::vector<std::string> symbols = {
    "X","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"};
const std::set<std::string> controls = {"Properties","Lattice","pbc","BXYZVersion","BXYZStorage",
    "BXYZPayloadBytes","BXYZRawBytes","BXYZCodec","BXYZFilter","BXYZChecksum"};

[[noreturn]] void fail(const std::string& message) { throw Error(message); }
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) fail("size overflow");
    return a + b;
}
std::size_t mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) fail("size overflow");
    return a * b;
}
bool space(char c) { return c==' ' || c=='\t' || c=='\r' || c=='\n'; }
bool name_ok(const std::string& s) {
    auto letter=[](char c) { return (c>='a'&&c<='z') || (c>='A'&&c<='Z') || c=='_'; };
    if (s.empty() || !letter(s[0])) return false;
    return std::all_of(s.begin(), s.end(), [&](char c) { return letter(c) || (c>='0'&&c<='9'); });
}
std::size_t integer(const std::string& s) {
    if (s.empty()) fail("missing nonnegative integer");
    std::size_t result=0;
    for (char c:s) { if(c<'0'||c>'9') fail("invalid nonnegative integer: " + s); result=add(mul(result,10), c-'0'); }
    return result;
}
std::string trimmed(std::string s) {
    const auto first=s.find_first_not_of(" \t\r\n");
    return first==std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n")-first+1);
}
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> result;
    std::size_t begin=0;
    for (;;) { auto end=s.find(delimiter,begin); result.push_back(s.substr(begin,end-begin));
        if(end==std::string::npos) return result; begin=end+1; }
}
// Validate Unicode without depending on locale or a platform-specific codec.
bool utf8(const std::string& s) {
    for(std::size_t i=0;i<s.size();) {
        auto c=static_cast<unsigned char>(s[i++]);
        if(c<128) continue;
        unsigned need=0, code=0, minimum=0;
        if(c>=0xc2 && c<=0xdf) {need=1;code=c&31;minimum=0x80;}
        else if(c>=0xe0 && c<=0xef) {need=2;code=c&15;minimum=0x800;}
        else if(c>=0xf0 && c<=0xf4) {need=3;code=c&7;minimum=0x10000;}
        else return false;
        if(need>s.size()-i) return false;
        while(need--) { auto d=static_cast<unsigned char>(s[i++]); if((d&0xc0)!=0x80) return false; code=(code<<6)|(d&63); }
        if(code<minimum || code>0x10ffff || (code>=0xd800 && code<=0xdfff)) return false;
    }
    return true;
}
struct Token { std::string value, raw; };
using Header = std::map<std::string, Token>;
Header parse_header(const std::string& text) {
    if(!utf8(text)) fail("invalid UTF-8 information line");
    Header result;
    std::size_t i=0;
    while(i<text.size()) {
        while(i<text.size() && space(text[i])) ++i;
        if(i==text.size()) break;
        auto start=i;
        while(i<text.size() && !space(text[i]) && text[i]!='=') ++i;
        auto key=text.substr(start,i-start);
        if(!name_ok(key)) fail("invalid metadata key: " + key);
        while(i<text.size() && space(text[i])) ++i;
        std::string value="T";
        if(i<text.size() && text[i]=='=') {
            ++i;
            while(i<text.size() && space(text[i])) ++i;
            value.clear();
            char close=0;
            if(i<text.size()) {
                char c=text[i];
                if(c=='\''||c=='"') close=c;
                else if(c=='[') close=']';
                else if(c=='{') close='}';
            }
            if(close) ++i;
            bool ended=!close;
            while(i<text.size()) {
                char c=text[i];
                if(c=='\\') { if(++i==text.size()) fail("unfinished metadata escape"); value+=text[i++]; }
                else if(close && c==close) {++i;ended=true;break;}
                else if(!close && space(c)) break;
                else {value+=c;++i;}
            }
            if(!ended) fail("unclosed metadata value");
            if(close && i<text.size() && !space(text[i])) fail("invalid text after quoted value");
        }
        if(!result.emplace(key,Token{value,trimmed(text.substr(start,i-start))}).second) fail("duplicate metadata key: " + key);
    }
    return result;
}
std::string get(const Header& h, const std::string& key) {
    auto it=h.find(key); if(it==h.end()) fail("missing " + key); return it->second.value;
}
std::string canonical(const std::string& name) {
    if(name=="species") return "symbols";
    if(name=="pos") return "positions";
    if(name=="Z") return "numbers";
    if(name=="charge") return "charges";
    return name;
}
struct Field { std::string name; char kind; std::size_t count; std::string storage; Values data; };
std::size_t width(const Field& f) {
    const auto& t=f.storage;
    if(t=="S") return 0;
    if(t=="Z8"||t=="L8") return 1;
    if(t=="R32"||t=="I32") return 4;
    if(t=="R64"||t=="I64") return 8;
    if(t.size()>1 && t[0]=='S') {auto w=integer(t.substr(1)); if(w) return w;}
    fail("invalid storage type: " + t);
}
std::optional<std::size_t> stride(const std::vector<Field>& fields) {
    std::size_t result=0;
    for(const auto& f:fields) {auto w=width(f); if(!w) return {}; result=add(result,mul(w,f.count));}
    return result;
}
void schema_check(const std::vector<Field>& fields, bool structure=false) {
    if(fields.empty()) fail("at least one property required");
    bool positions=false,species=false,numbers=false;
    std::set<std::string> names;
    for(const auto& f:fields) {
        if(!name_ok(f.name)||!f.count||!names.insert(canonical(f.name)).second) fail("invalid or duplicate property: " + f.name);
        if(f.name=="pos" && f.kind=='R' && f.count==3) positions=true;
        if(f.name=="species" && f.kind=='S' && f.count==1) species=true;
        if(f.name=="Z" && f.kind=='I' && f.count==1) numbers=true;
    }
    if(structure && (!positions||(!species&&!numbers))) fail("pos:R:3 and species:S:1 or Z:I:1 required");
}
std::vector<Field> parse_fields(const Header& h, int version) {
    auto props=split(get(h,"Properties"),':');
    if(props.size()%3) fail("invalid Properties triples");
    std::vector<std::string> storage;
    if(version>=2) storage=split(get(h,"BXYZStorage"),':');
    else if(h.count("BXYZStorage")) fail("storage metadata in version 1");
    if(version>=2 && storage.size()!=props.size()/3) fail("storage count mismatch");
    std::vector<Field> fields;
    for(std::size_t i=0;i<props.size();i+=3) {
        if(props[i+1].size()!=1 || std::string("RILS").find(props[i+1])==std::string::npos) fail("invalid logical type");
        char kind=props[i+1][0];
        auto token=version>=2 ? storage[i/3] : kind=='R'?"R64":kind=='I'?"I64":kind=='L'?"L8":"S";
        Field f{props[i],kind,integer(props[i+2]),token,{}};
        bool valid=kind=='R' ? token=="R32"||token=="R64" : kind=='I' ? token=="I32"||token=="I64" :
                   kind=='L' ? token=="L8" : token=="S" || (version>=3 && token.size()>1 && token[0]=='S') ||
                   (version>=3 && token=="Z8" && f.name=="species" && f.count==1);
        if(!valid) fail("storage type disagrees with property: " + f.name);
        width(f); fields.push_back(std::move(f));
    }
    schema_check(fields); return fields;
}

std::string sha256(const std::string& prefix, const Bytes& bytes) {
    // SHA-256's streaming ABI is also available on the older compute-node OpenSSL.
    SHA256_CTX ctx;
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if(SHA256_Init(&ctx)!=1 || SHA256_Update(&ctx,prefix.data(),prefix.size())!=1 ||
       SHA256_Update(&ctx,bytes.data(),bytes.size())!=1 || SHA256_Final(digest,&ctx)!=1)
        fail("SHA-256 calculation failed");
    const unsigned length=SHA256_DIGEST_LENGTH;
    std::string hex; const char* alphabet="0123456789abcdef";
    for(unsigned i=0;i<length;++i) {hex+=alphabet[digest[i]>>4];hex+=alphabet[digest[i]&15];}
    return hex;
}
void append_uint(Bytes& bytes, std::uint64_t value, std::size_t size) {
    for(std::size_t i=0;i<size;++i) {bytes.push_back(static_cast<std::uint8_t>(value&255));value>>=8;}
}
std::uint64_t bits(double value) {std::uint64_t result;std::memcpy(&result,&value,8);return result;}
struct Cursor {
    const Bytes& bytes; std::size_t offset=0;
    void require(std::size_t size) const {if(size>bytes.size()-offset) fail("truncated binary field");}
    std::uint64_t uint(std::size_t size) {
        require(size); std::uint64_t value=0;
        for(std::size_t i=0;i<size;++i) value|=static_cast<std::uint64_t>(bytes[offset++])<<(8*i);
        return value;
    }
    std::string text(std::size_t size) {
        require(size);std::string s(reinterpret_cast<const char*>(bytes.data()+offset),size);offset+=size;return s;
    }
};

Bytes shuffle(const Bytes& raw,std::size_t w,bool inverse) {
    if(!w || raw.size()%w) fail("invalid shuffle dimensions");
    Bytes result(raw.size()); auto n=raw.size()/w;
    for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<w;++j)
        if(inverse) result[i*w+j]=raw[j*n+i]; else result[j*n+i]=raw[i*w+j];
    return result;
}
void zcheck(std::size_t code) {if(ZSTD_isError(code)) fail(std::string("Zstd: ")+ZSTD_getErrorName(code));}
Bytes compress(const Bytes& raw,int level) {
    Bytes result(ZSTD_compressBound(raw.size()));
    auto size=ZSTD_compress(result.data(),result.size(),raw.data(),raw.size(),level);zcheck(size);result.resize(size);return result;
}
Bytes decompress(const Bytes& packed,std::size_t size) {
    auto content=ZSTD_getFrameContentSize(packed.data(),packed.size());
    if(content==ZSTD_CONTENTSIZE_ERROR || content==ZSTD_CONTENTSIZE_UNKNOWN || content!=size) fail("Zstd content size mismatch");
    auto compressed_size=ZSTD_findFrameCompressedSize(packed.data(),packed.size());zcheck(compressed_size);
    if(compressed_size!=packed.size()) fail("extra data after Zstd frame");
    Bytes result(size);
    auto actual=ZSTD_decompress(result.data(),result.size(),packed.data(),packed.size());zcheck(actual);
    if(actual!=size) fail("decompressed length mismatch"); return result;
}

std::vector<Field> prepare(const Frame& frame,const Options& opts) {
    if(frame.positions.size()!=mul(frame.size(),3)) fail("positions must contain 3*N values");
    for(auto z:frame.numbers) if(z>=symbols.size()) fail("invalid atomic number");
    std::vector<Field> fields{{"species",'S',1,"Z8",frame.numbers},{"pos",'R',3,"",frame.positions}};
    for(const auto& p:frame.properties) {
        char kind=std::holds_alternative<std::vector<double>>(p.values)?'R':
                  std::holds_alternative<std::vector<std::int64_t>>(p.values)?'I':
                  std::holds_alternative<Bytes>(p.values)?'L':'S';
        fields.push_back(Field{p.name,kind,p.components,"",p.values});
    }
    schema_check(fields, true);
    for(auto& f:fields) {
        if(std::visit([](const auto& v){return v.size();},f.data)!=mul(frame.size(),f.count)) fail("array size mismatch: "+f.name);
        if(f.storage=="Z8") continue;
        if(f.kind=='R') {
            bool fits=true;
            for(double x:std::get<std::vector<double>>(f.data)) {
                if(std::isnan(x)) {fits=false;break;}
                // Avoid out-of-range floating conversion, which C++ does not guarantee.
                if(std::isfinite(x) && std::abs(x)>std::numeric_limits<float>::max()) {fits=false;break;}
                double back=static_cast<double>(static_cast<float>(x));
                if(opts.float_rtol==0 && opts.float_atol==0) fits=bits(x)==bits(back);
                else if(std::isfinite(x)) fits=std::abs(back-x)<=opts.float_atol+opts.float_rtol*std::abs(x);
                if(!fits) break;
            }
            f.storage=fits?"R32":"R64";
        } else if(f.kind=='I') {
            const auto& v=std::get<std::vector<std::int64_t>>(f.data);
            bool fits=std::all_of(v.begin(),v.end(),[](auto x){return x>=INT32_MIN&&x<=INT32_MAX;});
            f.storage=fits?"I32":"I64";
        } else if(f.kind=='L') {
            for(auto b:std::get<Bytes>(f.data)) if(b>1) fail("logical values must be 0 or 1");
            f.storage="L8";
        } else {
            std::size_t w=1,total=0;bool nul=false;
            const auto& values=std::get<std::vector<std::string>>(f.data);
            for(const auto& s:values) {if(!utf8(s)) fail("invalid UTF-8 string");w=std::max(w,s.size());total=add(total,s.size());nul|=s.find('\0')!=std::string::npos;}
            if(nul && opts.string_mode==StringMode::fixed) fail("fixed strings cannot contain NUL");
            bool fixed=opts.string_mode==StringMode::fixed || (opts.string_mode==StringMode::automatic && !nul && mul(w,values.size())<=add(total,mul(4,values.size())));
            f.storage=fixed?"S"+std::to_string(w):"S";
        }
    }
    return fields;
}
Bytes encode(const std::vector<Field>& fields,std::size_t n) {
    Bytes raw;
    if(auto w=stride(fields)) raw.reserve(mul(n,*w));
    for(std::size_t atom=0;atom<n;++atom) for(const auto& f:fields) for(std::size_t c=0;c<f.count;++c) {
        auto i=atom*f.count+c, w=width(f);
        if(f.storage=="Z8"||f.kind=='L') raw.push_back(std::get<Bytes>(f.data)[i]);
        else if(f.kind=='R') {
            double x=std::get<std::vector<double>>(f.data)[i];
            if(w==8) append_uint(raw,bits(x),8);
            else {float v=static_cast<float>(x);std::uint32_t b;std::memcpy(&b,&v,4);append_uint(raw,b,4);}
        } else if(f.kind=='I') append_uint(raw,static_cast<std::uint64_t>(std::get<std::vector<std::int64_t>>(f.data)[i]),w);
        else {
            const auto& s=std::get<std::vector<std::string>>(f.data)[i];
            if(!w) {if(s.size()>UINT32_MAX) fail("string exceeds uint32 length");append_uint(raw,s.size(),4);}
            raw.insert(raw.end(),s.begin(),s.end());
            if(w) raw.insert(raw.end(),w-s.size(),0);
        }
    }
    return raw;
}
void decode_value(Cursor& cursor,Field& f,std::size_t i) {
    auto w=width(f);
    if(f.storage=="Z8"||f.kind=='L') {
        auto v=cursor.uint(1);if(v>(f.storage=="Z8"?118u:1u)) fail("invalid atomic number or logical value");
        std::get<Bytes>(f.data)[i]=static_cast<std::uint8_t>(v);
    } else if(f.kind=='R') {
        double x;
        if(w==8) {auto b=cursor.uint(8);std::memcpy(&x,&b,8);}
        else {auto b=static_cast<std::uint32_t>(cursor.uint(4));float v;std::memcpy(&v,&b,4);x=v;}
        std::get<std::vector<double>>(f.data)[i]=x;
    } else if(f.kind=='I') {
        auto value=cursor.uint(w);std::int64_t x;
        if(w==4) {auto u=static_cast<std::uint32_t>(value);std::int32_t v;std::memcpy(&v,&u,4);x=v;}
        else std::memcpy(&x,&value,8);
        std::get<std::vector<std::int64_t>>(f.data)[i]=x;
    } else {
        auto s=cursor.text(w?w:static_cast<std::size_t>(cursor.uint(4)));
        if(w) {auto end=s.find('\0');if(end!=std::string::npos) {
            if(s.find_first_not_of('\0',end)!=std::string::npos) fail("invalid fixed string padding");s.resize(end);}}
        if(!utf8(s)) fail("invalid UTF-8 string");std::get<std::vector<std::string>>(f.data)[i]=std::move(s);
    }
}
void decode(Bytes const& raw,std::vector<Field>& fields,std::size_t n,int version) {
    std::size_t minimum=version==2?1:0;
    for(const auto& f:fields) minimum=add(minimum,mul(f.count,width(f)?width(f):4));
    if(mul(n,minimum)>raw.size()) fail("truncated atom records");
    for(auto& f:fields) {
        auto size=mul(n,f.count);
        if(f.storage=="Z8"||f.kind=='L') f.data=Bytes(size);
        else if(f.kind=='R') f.data=std::vector<double>(size);
        else if(f.kind=='I') f.data=std::vector<std::int64_t>(size);
        else f.data=std::vector<std::string>(size);
    }
    Cursor cursor{raw};
    if(version==1) {
        for(auto& f:fields) for(std::size_t i=0;i<mul(n,f.count);++i) decode_value(cursor,f,i);
    } else {
        for(std::size_t a=0;a<n;++a) {
            for(auto& f:fields) for(std::size_t c=0;c<f.count;++c) decode_value(cursor,f,a*f.count+c);
            if(version==2 && cursor.uint(1)!=10) fail("missing LF after atom "+std::to_string(a+1));
        }
    }
    if(cursor.offset!=raw.size()) fail("extra bytes in atom payload");
}
Frame assemble(std::vector<Field>& fields,const Header& header,std::size_t n) {
    schema_check(fields, true);
    Frame result;std::optional<Bytes> from_symbols,from_numbers;
    for(auto& f:fields) {
        if(f.name=="species") {
            if(f.storage=="Z8") from_symbols=std::get<Bytes>(std::move(f.data));
            else {Bytes z;z.reserve(n);for(const auto& symbol:std::get<std::vector<std::string>>(f.data)) {
                auto it=std::find(symbols.begin(),symbols.end(),symbol);if(it==symbols.end()) fail("unknown element: "+symbol);
                z.push_back(static_cast<std::uint8_t>(it-symbols.begin()));}from_symbols=std::move(z);}
        } else if(f.name=="Z") {
            Bytes z;z.reserve(n);for(auto x:std::get<std::vector<std::int64_t>>(f.data)) {
                if(x<0||x>118) fail("invalid atomic number");z.push_back(static_cast<std::uint8_t>(x));}from_numbers=std::move(z);
        } else if(f.name=="pos") result.positions=std::get<std::vector<double>>(std::move(f.data));
        else result.properties.push_back(Property{f.name,f.count,std::move(f.data)});
    }
    if(from_symbols && from_numbers && *from_symbols!=*from_numbers) fail("species and Z disagree");
    result.numbers=from_symbols?std::move(*from_symbols):std::move(*from_numbers);
    if(auto it=header.find("Lattice");it!=header.end()) {
        std::istringstream values(it->second.value);values.imbue(std::locale::classic());
        for(auto& x:result.cell) if(!(values>>x)) fail("invalid Lattice");
        std::string extra;if(values>>extra) fail("Lattice must have 9 values");result.pbc={true,true,true};
    }
    if(auto it=header.find("pbc");it!=header.end()) {
        std::istringstream values(it->second.value);std::vector<bool> flags;std::string token;
        while(values>>token) {if(token=="T"||token=="True"||token=="true"||token=="TRUE") flags.push_back(true);
            else if(token=="F"||token=="False"||token=="false"||token=="FALSE") flags.push_back(false);else fail("invalid pbc");}
        if(flags.size()!=1&&flags.size()!=3) fail("pbc requires one or three booleans");
        for(std::size_t i=0;i<3;++i) result.pbc[i]=flags[flags.size()==1?0:i];
    }
    for(const auto& kv:header) if(!controls.count(kv.first)) {
        for(const auto& f:fields) if(canonical(kv.first)==canonical(f.name)) fail("metadata/array name conflict");
        if(!result.info_extxyz.empty()) result.info_extxyz+=' ';result.info_extxyz+=kv.second.raw;
    }
    return result;
}
std::string make_header(const Frame& frame,const std::vector<Field>& fields) {
    if(frame.info_extxyz.find_first_of("\r\n")!=std::string::npos) fail("metadata must be one line");
    for(const auto& kv:parse_header(frame.info_extxyz)) {
        if(controls.count(kv.first)) fail("reserved metadata key: "+kv.first);
        for(const auto& f:fields) if(canonical(kv.first)==canonical(f.name)) fail("metadata/array name conflict");
    }
    std::ostringstream h;h.imbue(std::locale::classic());h<<std::setprecision(17)<<"Lattice=\"";
    for(std::size_t i=0;i<9;++i) {if(i)h<<' ';h<<frame.cell[i];}h<<"\" Properties=";
    for(std::size_t i=0;i<fields.size();++i) {const auto& f=fields[i];if(i)h<<':';h<<f.name<<':'<<f.kind<<':'<<f.count;}
    h<<" pbc=\""<<(frame.pbc[0]?'T':'F')<<' '<<(frame.pbc[1]?'T':'F')<<' '<<(frame.pbc[2]?'T':'F')<<'"';
    if(!frame.info_extxyz.empty()) h<<' '<<frame.info_extxyz;
    h<<" BXYZStorage=";
    for(std::size_t i=0;i<fields.size();++i) {if(i)h<<':';h<<fields[i].storage;}
    return h.str();
}

std::vector<Field> prepare(const RawFrame& frame,const Options& opts) {
    std::vector<Field> fields;
    for(const auto& p:frame.properties) {
        char kind=std::holds_alternative<std::vector<double>>(p.values)?'R':
                  std::holds_alternative<std::vector<std::int64_t>>(p.values)?'I':
                  std::holds_alternative<Bytes>(p.values)?'L':'S';
        fields.push_back(Field{p.name,kind,p.components,"",p.values});
        auto& f=fields.back();
        if(f.name=="species" && kind=='S' && f.count==1) {
            Bytes numbers; bool known=true;
            for(const auto& symbol:std::get<std::vector<std::string>>(f.data)) {
                auto it=std::find(symbols.begin(),symbols.end(),symbol);
                if(it==symbols.end()) {known=false;break;}
                numbers.push_back(static_cast<std::uint8_t>(it-symbols.begin()));
            }
            if(known) {f.storage="Z8";f.data=std::move(numbers);}
        }
    }
    schema_check(fields);
    for(auto& f:fields) {
        if(std::visit([](const auto& v){return v.size();},f.data)!=mul(frame.count,f.count)) fail("array size mismatch: "+f.name);
        if(f.storage=="Z8") continue;
        if(f.kind=='R') {
            bool fits=true;
            for(double x:std::get<std::vector<double>>(f.data)) {
                if(std::isnan(x)) {fits=false;break;}
                // Avoid out-of-range floating conversion, which C++ does not guarantee.
                if(std::isfinite(x) && std::abs(x)>std::numeric_limits<float>::max()) {fits=false;break;}
                double back=static_cast<double>(static_cast<float>(x));
                if(opts.float_rtol==0 && opts.float_atol==0) fits=bits(x)==bits(back);
                else if(std::isfinite(x)) fits=std::abs(back-x)<=opts.float_atol+opts.float_rtol*std::abs(x);
                if(!fits) break;
            }
            f.storage=fits?"R32":"R64";
        } else if(f.kind=='I') {
            const auto& v=std::get<std::vector<std::int64_t>>(f.data);
            bool fits=std::all_of(v.begin(),v.end(),[](auto x){return x>=INT32_MIN&&x<=INT32_MAX;});
            f.storage=fits?"I32":"I64";
        } else if(f.kind=='L') {
            for(auto b:std::get<Bytes>(f.data)) if(b>1) fail("logical values must be 0 or 1");
            f.storage="L8";
        } else {
            std::size_t w=1,total=0;bool nul=false;
            const auto& values=std::get<std::vector<std::string>>(f.data);
            for(const auto& s:values) {if(!utf8(s)) fail("invalid UTF-8 string");w=std::max(w,s.size());total=add(total,s.size());nul|=s.find('\0')!=std::string::npos;}
            if(nul && opts.string_mode==StringMode::fixed) fail("fixed strings cannot contain NUL");
            bool fixed=opts.string_mode==StringMode::fixed || (opts.string_mode==StringMode::automatic && !nul && mul(w,values.size())<=add(total,mul(4,values.size())));
            f.storage=fixed?"S"+std::to_string(w):"S";
        }
    }
    return fields;
}

std::string make_header(const RawFrame& frame,const std::vector<Field>& fields) {
    if(frame.info_extxyz.find_first_of("\r\n")!=std::string::npos) fail("metadata must be one line");
    for(const auto& kv:parse_header(frame.info_extxyz))
        if(controls.count(kv.first) && kv.first!="Lattice" && kv.first!="pbc") fail("reserved metadata key: "+kv.first);
    std::ostringstream h;h.imbue(std::locale::classic());
    if(!frame.info_extxyz.empty()) h<<frame.info_extxyz<<' ';
    h<<"Properties=";
    for(std::size_t i=0;i<fields.size();++i) {const auto& f=fields[i];if(i)h<<':';h<<f.name<<':'<<f.kind<<':'<<f.count;}
    h<<" BXYZStorage=";
    for(std::size_t i=0;i<fields.size();++i) {if(i)h<<':';h<<fields[i].storage;}
    return h.str();
}

RawFrame raw_frame(std::vector<Field>& fields,const Header& header,std::size_t count) {
    RawFrame result;result.count=count;
    for(auto& f:fields) {
        if(f.storage=="Z8") {
            std::vector<std::string> labels;
            for(auto z:std::get<Bytes>(f.data)) labels.push_back(symbols.at(z));
            f.data=std::move(labels);
        }
        result.properties.push_back({f.name,f.count,std::move(f.data)});
    }
    for(const auto& kv:header) if(!controls.count(kv.first)||kv.first=="Lattice"||kv.first=="pbc") {
        if(!result.info_extxyz.empty()) result.info_extxyz+=' ';
        result.info_extxyz+=kv.second.raw;
    }
    return result;
}

bool line(std::istream& stream,std::string& s) {
    s.clear();char c;
    while(stream.get(c)) {s+=c;if(c=='\n') return true;}
    if(!s.empty()) fail("truncated text header");
    if(stream.bad()) fail("file read failed");return false;
}
} // namespace

std::string element_symbol(std::uint8_t number) { return symbols.at(number); }

Frame as_structure(const RawFrame& raw) {
    auto fields=prepare(raw,Options{});
    return assemble(fields,parse_header(raw.info_extxyz),raw.count);
}

Reader::Reader(const std::filesystem::path& path,bool require_checksum)
    :stream_(path,std::ios::binary),require_checksum_(require_checksum) {
    if(!stream_) fail("cannot open input file");file_size_=std::filesystem::file_size(path);
}
bool Reader::advance(RawFrame* output) {
    try {
        std::string first,second;
        if(!line(stream_,first)) return false;
        if(!line(stream_,second)) fail("missing information line");
        auto n=integer(trimmed(first));auto h=parse_header(second);
        auto ver=integer(get(h,"BXYZVersion"));if(ver<1||ver>5) fail("unsupported BXYZVersion");
        int version=static_cast<int>(ver);auto length=integer(get(h,"BXYZPayloadBytes"));
        auto fields=parse_fields(h,version);auto w=stride(fields);
        std::string prefix,digest;
        if(version>=4) {
            const std::string marker=" BXYZChecksum=sha256:";
            auto at=second.rfind(marker);auto nl=second.size()>=2 && second[second.size()-2]=='\r'?2u:1u;
            if(at==std::string::npos || second.size()-at!=marker.size()+64+nl) fail("missing or malformed SHA-256 frame checksum");
            digest=second.substr(at+marker.size(),64);
            if(!std::all_of(digest.begin(),digest.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');})) fail("malformed checksum digest");
            prefix=first+second.substr(0,at)+second.substr(second.size()-nl);
        } else if(h.count("BXYZChecksum")) fail("checksum requires version 4 or 5");
        else if(require_checksum_) fail("legacy frame has no checksum");
        std::size_t raw_size=length;std::string filter;
        if(version==5) {
            raw_size=integer(get(h,"BXYZRawBytes"));filter=get(h,"BXYZFilter");
            if(get(h,"BXYZCodec")!="zstd" || (filter!="none"&&filter!="shuffle")) fail("unsupported compression codec/filter");
            if(filter=="shuffle"&&!w) fail("shuffle requires fixed records");
            if(w&&raw_size!=mul(n,*w)) fail("raw byte count mismatch");
        } else if(h.count("BXYZCodec")||h.count("BXYZFilter")||h.count("BXYZRawBytes")) fail("compression metadata requires version 5");
        auto position=stream_.tellg();
        if(position<0 || static_cast<std::uint64_t>(position)>file_size_ || length>file_size_-static_cast<std::uint64_t>(position)) fail("truncated binary payload");
        if(length>static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) fail("frame too large for stream");
        if(!output) {stream_.seekg(static_cast<std::streamoff>(length),std::ios::cur);if(!stream_)fail("seek failed");++index_;return true;}
        Bytes payload(length);
        if(length) stream_.read(reinterpret_cast<char*>(payload.data()),static_cast<std::streamsize>(length));
        if(!stream_) fail("truncated binary payload");
        if(version>=4 && sha256(prefix,payload)!=digest) fail("SHA-256 frame checksum mismatch");
        if(version==5) {payload=decompress(payload,raw_size);if(filter=="shuffle")payload=shuffle(payload,*w,true);}
        decode(payload,fields,n,version);auto frame=raw_frame(fields,h,n);*output=std::move(frame);++index_;return true;
    } catch(const std::exception& e) {throw Error("frame "+std::to_string(index_+1)+": "+e.what());}
}
bool Reader::next_raw(RawFrame& frame) {return advance(&frame);}
bool Reader::next(Frame& frame) {
    RawFrame raw;if(!next_raw(raw)) return false;
    try {
        auto fields=prepare(raw,Options{});
        frame=assemble(fields,parse_header(raw.info_extxyz),raw.count);
    } catch(const std::exception& e) {throw Error("frame "+std::to_string(index_)+": "+e.what());}
    return true;
}
RawFrame Reader::read_raw(std::size_t index) {
    stream_.clear();stream_.seekg(0);index_=0;
    for(std::size_t i=0;i<index;++i) if(!advance(nullptr)) fail("frame index out of range");
    RawFrame frame;if(!next_raw(frame)) fail("frame index out of range");return frame;
}
bool Reader::scan(std::uint64_t& offset) {
    auto position=stream_.tellg();
    if(position<0) {if(stream_.eof()) return false;fail("cannot obtain frame offset");}
    if(!advance(nullptr)) return false;
    offset=static_cast<std::uint64_t>(position);return true;
}
Frame Reader::read_at(std::uint64_t offset,std::size_t index) {
    if(offset>=file_size_ || offset>static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) fail("frame offset out of range");
    stream_.clear();stream_.seekg(static_cast<std::streamoff>(offset));index_=index;
    if(!stream_) fail("seek failed");
    Frame frame;if(!next(frame)) fail("frame offset out of range");return frame;
}
Frame Reader::read(std::size_t index) {
    stream_.clear();stream_.seekg(0);index_=0;
    for(std::size_t i=0;i<index;++i) if(!advance(nullptr)) fail("frame index out of range");
    Frame frame;if(!next(frame)) fail("frame index out of range");return frame;
}
Writer::Writer(const std::filesystem::path& path,Options options,bool append):options_(options) {
    if(!std::isfinite(options.float_rtol)||!std::isfinite(options.float_atol)||options.float_rtol<0||options.float_atol<0) fail("invalid floating tolerance");
    if(options.compression_level<1||options.compression_level>22) fail("compression level must be 1-22");
    auto suffix=path.extension().string();std::transform(suffix.begin(),suffix.end(),suffix.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
    compressed_=options.compressed.value_or(suffix==".zbxyz");
    if(append && std::filesystem::exists(path)) {Reader reader(path);RawFrame frame;while(reader.next_raw(frame))++index_;}
    stream_.open(path,std::ios::binary|(append?std::ios::app:std::ios::trunc));if(!stream_) fail("cannot open output file");
}
void Writer::write(const Frame& frame) {
    auto fields=prepare(frame,options_);
    auto header=parse_header(make_header(frame,fields));
    auto raw=raw_frame(fields,header,frame.size());
    write(raw);
}
void Writer::write(const RawFrame& frame) {
    if(failed_) fail("writer is unusable after an output failure");
    try {
        auto fields=prepare(frame,options_);auto header=make_header(frame,fields);auto payload=encode(fields,frame.count);
        if(compressed_) {
            auto raw_size=payload.size();auto w=stride(fields);
            if(w) payload=shuffle(payload,*w,false);
            payload=compress(payload,options_.compression_level);
            header+=" BXYZCodec=zstd BXYZFilter="+std::string(w?"shuffle":"none")+" BXYZRawBytes="+std::to_string(raw_size);
        }
        header+=" BXYZVersion="+std::to_string(compressed_?5:4)+" BXYZPayloadBytes="+std::to_string(payload.size());
        auto count=std::to_string(frame.count)+"\n";
        auto digest=sha256(count+header+"\n",payload);
        auto prefix=count+header+" BXYZChecksum=sha256:"+digest+"\n";
        if(payload.size()>static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) fail("frame too large for stream");
        stream_.write(prefix.data(),static_cast<std::streamsize>(prefix.size()));
        if(!payload.empty()) stream_.write(reinterpret_cast<const char*>(payload.data()),static_cast<std::streamsize>(payload.size()));
        if(!stream_) {failed_=true;fail("frame write failed");}++index_;
    } catch(const std::exception& e) {throw Error("frame "+std::to_string(index_+1)+": "+e.what());}
}
void Writer::flush() {stream_.flush();if(!stream_) {failed_=true;fail("flush failed");}}
std::size_t verify(const std::filesystem::path& path) {Reader reader(path,true);RawFrame frame;std::size_t n=0;while(reader.next_raw(frame))++n;return n;}
} // namespace binary_xyz
