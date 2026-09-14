# 原生 C++ 支持

提供 C++17 库，无 Python 运行时依赖。使用 Zstd 压缩、OpenSSL Crypto 计算 SHA-256，文件格式与 Python 实现互通。写入支持未压缩 v4 和压缩 v5；读取支持 v1–v5。原子结构在内存中由 `Frame` 表示，`Writer` 可持续打开文件供 MD 引擎逐帧写入，`Reader` 支持逐帧读取和跳帧。

重复随机读取时可先用 `Reader::scan(offset)` 逐帧获取字节偏移，再用 `Reader::read_at(offset, index)` 直接读取并校验目标帧。扫描只验证帧头及负载边界，不解码或校验跳过的负载；偏移只适用于同一未改动文件。`read_at` 后的 `next` 从下一帧继续。第二个参数为从 0 开始的帧编号，用于错误报告。

## 构建

需要 CMake 3.18+、C++17 编译器、Zstd 和 OpenSSL 的开发头文件及库。纯 C++ 构建关闭 BUILD_TESTING，不要求安装 Python。

```sh
cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build/cpp --config Release
```

本机 Windows 环境已使用 MSVC 19.35、Zstd、OpenSSL 3.5.6 验证。对应 PowerShell 命令为：

```powershell
cmake -S . -B build/cpp -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=D:/anaconda3/Library -DPython3_EXECUTABLE=D:/anaconda3/python.exe
cmake --build build/cpp --config Release
$env:PATH = 'D:\anaconda3\Library\bin;' + $env:PATH
ctest --test-dir build/cpp -C Release --output-on-failure
```

动态链接的 Zstd 和 OpenSSL DLL 必须位于运行时搜索路径；上述路径仅适用于本机，应换成实际依赖位置。自动测试需要 Python、pytest 和本项目的 Python 依赖，包含原生 API 测试及 Python/C++ 双向互读测试。其他平台采用相同 CMake 项目，但尚未在本机之外执行构建。

## 接入 MD 引擎

```cpp
#include <binary_xyz/binary_xyz.hpp>

binary_xyz::Options options;
options.compression_level = 3;
binary_xyz::Writer trajectory("trajectory.zbxyz", options);

binary_xyz::Frame frame;
frame.numbers = {22, 8}; // Ti, O: uint8 原子序数，0=X，1–118为元素
frame.positions = {0.1, 0.2, 0.3, 1.1, 1.2, 1.3};
frame.cell = {5, 0, 0, 0, 5, 0, 0, 0, 5};
frame.pbc = {true, true, true};
frame.properties.push_back({"forces", 3,
    std::vector<double>{0.1, 0, 0, -0.1, 0, 0}});

for (int step = 0; step < 100; ++step) {
    // 用 MD 引擎当前帧更新 positions、forces、cell 等。
    frame.info_extxyz = "step=" + std::to_string(step) + " energy=-1.25";
    trajectory.write(frame);
}
trajectory.flush(); // 显式检测缓冲输出错误；也可定期调用。
```

可运行示例位于 `cpp/examples/md_writer.cpp`。每个 Frame 独立编码、重排、压缩及校验，完整编码完成后才写入帧头与负载。不积累整条轨迹，但会有当前帧的数据副本及压缩缓冲。`write()` 不扫描或读取此前的帧。`Writer(path, options, true)` 表示追加，只在构造时验证已有文件一次；不要在每个 MD 步骤重新构造 Writer。

`Options.compressed` 默认根据 `.zbxyz` 后缀选择，显式 true/false 可覆盖。默认 `float_rtol=1e-7`、`float_atol=0`，允许和 Python 相同的小幅舍入；两者设为零启用严格无损位宽选择。整数无损选择 int32/int64。`string_mode` 为 `automatic`、`fixed` 或 `variable`；元素始终使用 Z8，普通字符串按 UTF-8 字节数选择固定宽度或长度前缀。固定字符串不允许自身含 NUL，自动模式会转为变长。

`Frame.properties` 中属性名使用 extxyz 名称，分量按原子顺序展平，长度必须等于 N×components。类型映射为 `vector<double>`→R、`vector<int64_t>`→I、`vector<uint8_t>`→L（仅0/1）、`vector<string>`→S。元素编号和位置使用 Frame 的专用成员，不能在 properties 中重复。所有数组由 Frame 拥有；当前不提供直接引用 MD 内存的零拷贝接口。

`cell` 为行优先的9个值，与 ASE 晶胞矩阵相同。额外元数据 `info_extxyz` 是一行 extxyz 键值文本；可存能量、应力、时间步以及 ASE `_JSON` 编码值。读取时保留非系统字段的原始值表达式，不把它们强制转换成 C++ JSON 对象。字符串转义和数值单位遵循 extxyz，元数据不参与浮点位宽选择。系统字段、重复属性及元数据与属性的同名冲突会报错。

## 读取和校验

```cpp
binary_xyz::Reader reader("trajectory.zbxyz");
binary_xyz::Frame frame;
while (reader.next(frame)) {
    // 处理一个已校验、解压和解码的 Frame。
}
auto ninth = reader.read(8); // 重置游标，跳过前8帧负载，只解码目标帧
auto frames = binary_xyz::verify("trajectory.zbxyz");
```

帧索引从0开始，错误中的帧号从1开始，异常类型为 `binary_xyz::Error`。`next()` 和 `read()` 验证实际读取的 v4/v5 帧，先验 SHA-256 再解压。被跳过帧只检查头部和文件边界，不检查其负载。`verify()` 逐帧验证并拒绝无校验的旧版帧；`Reader(path, true)` 同样要求校验和。`read(i)` 成功后，下一次 next() 从第 i+1 帧继续。旧版帧和不同原子数、属性集合的帧可以混合。

直接 Writer 与 Python write 一样，默认会截断现有目标；写入 I/O 错误可能留下不完整末帧，之后该 Writer 拒绝继续写入。格式校验异常发生在写入前，可修正输入后继续使用。析构关闭文件，但不会抛出关闭错误；需要可靠获知输出状态时显式 flush。刷新缓冲不等于磁盘断电持久性保证。不支持多个线程/进程并发写同一轨迹。

## CMake 集成和命令行

作为子项目使用：

```cmake
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(path/to/binary_xyz)
target_link_libraries(my_md PRIVATE binary_xyz::binary_xyz)
```

也可以先安装，再用 `find_package(binary_xyz CONFIG REQUIRED)` 和同一个导出目标：

```sh
cmake --install build/cpp --config Release --prefix path/to/install
```

安装包依然需要 Zstd、OpenSSL 开发依赖。示例消费项目在 `cpp/tests/consumer`。

```console
bxyz_cpp verify trajectory.zbxyz
bxyz_cpp inspect trajectory.zbxyz 8
bxyz_cpp convert input.bxyz output.zbxyz 3
bxyz_md_example example.zbxyz
```

C++ 命令行转换仅处理 BXYZ 二进制版本之间的转换，输出后缀为 .bxyz 或 .zbxyz，默认拒绝覆盖，先写临时文件再以硬链接发布，因此文件系统必须支持硬链接。此转换不会额外降低现有数值精度。文本 XYZ 转换仍使用 Python 命令。C++ API 提供数组和原始帧元数据；不依赖 ASE，也不构造计算器或约束对象。
