# Binary XYZ 格式规范

未压缩写入使用 v4，压缩写入使用 v5。读取支持 v1–v5 及混合轨迹。

## v5 独立帧压缩

扩展名为 .zbxyz。v5 的解压后原子数据编码与 v3/v4 完全相同。每帧头增加 BXYZCodec=zstd、BXYZFilter=shuffle 或 none，以及非负十进制 BXYZRawBytes（解压后的原始负载长度）。BXYZPayloadBytes 是压缩负载长度，仍可直接用于 seek 跳帧。上述字段均为保留键，旧版本不得出现。

BXYZFilter=shuffle 仅用于所有属性定长的帧。设 N 个原子、记录宽度 W，将原始负载视为 N×W 的 uint8 矩阵，转置后按行序列化，再用 Zstd 压缩。解压后视为 W×N 矩阵，转置恢复原始记录。无对齐填充。N=0 时解压内容为空，压缩负载仍是合法的空 Zstd 帧。含变长字符串的帧使用 none。

每个 BXYZ 帧的负载严格包含一个独立 Zstd 帧，不使用字典或依赖相邻帧；Zstd 帧必须携带已知内容长度，且与 BXYZRawBytes 相同。读取拒绝额外的压缩数据、解压错误和长度不符。定长布局还须满足 BXYZRawBytes=N×W。

沿用 v4 的 SHA-256 规则，但哈希覆盖压缩负载。读取顺序为头部检查、读取压缩负载、验证校验和、解压、逆重排、解码原子数据。跳帧只解析头部和检查文件边界，不解压或验证被跳过的负载。逐帧独立布局允许二进制拼接，没有文件级索引或终止块。



## v4 SHA-256 帧校验

v4 负载编码与 v3 完全相同。使用 BXYZVersion=4，第二行末尾在换行前追加一个字段：` BXYZChecksum=sha256:<64位小写十六进制摘要>`。字段必须仅出现一次，使用一个 ASCII 空格引出，摘要后立即是 LF 或 CRLF。字段为保留键，不得用于用户元数据。

计算方式：从原始第二行末尾移除引导空格及完整校验字段，但保留原来的换行字节；将原始第一行（含换行）、处理后的第二行（含换行）、完整负载依次输入 SHA-256。哈希的是原始字节，不对文本做规范化。写入器输出 LF，读取器也支持正确计算摘要的 CRLF 文件。BXYZPayloadBytes 仍只计算二进制负载，不包含帧头或校验字段。

读取器先解析和检查帧头，读取指定长度的负载，然后验证摘要，通过后才能解码并返回帧。缺失摘要、未知算法、摘要格式错误、重复字段或摘要不匹配均须报错并包含帧号。不得把校验失败的 v4 降级为旧版读取。v1–v3 出现 BXYZChecksum 时拒绝读取。

逐帧校验不保证整个轨迹未被删除、重排或重复整帧；无外部预期帧数时也无法识别整文件截断恰好发生在帧边界。SHA-256 在此用作完整性校验，不是带密钥的认证。

## v3 兼容布局

## v3 紧凑原子记录

使用 BXYZVersion=3，保留两行文本头与 BXYZPayloadBytes。原子记录按 Properties 顺序交错保存，无填充对齐，也无记录末尾 LF。第二行 LF 后直接开始负载，负载结束后直接开始下一帧。N=0 时负载为空。

浮点和整数沿用 v2 的自适应位宽及容差。BXYZStorage 与 Properties 中每个属性一一对应，新增两类标记：

| 标记 | 合法属性 | 编码 |
| --- | --- | --- |
| Z8 | 仅 species:S:1 | uint8 原子序数，0=X，1=H，8=O，22=Ti，118=Og；119–255非法 |
| S<W> | 任意 S 类型 | 正整数 W 表示每个标量的 UTF-8 字节宽度；不足部分右侧补 NUL，无长度前缀 |

S 标记仍表示原来的 uint32 长度前缀＋UTF-8 内容。S<W> 内不允许非尾部 NUL；例如 S6 可容纳两个常见汉字。W 最小为1，不能用 S0。空字符串用 W 个 NUL 表示。需要保存字符串本身的 NUL 时，必须使用变长 S。多分量字符串的每个分量各占 W 字节。

默认写入策略将 species 编码为 Z8。普通字符串逐帧逐属性比较两种负载长度，选择更小者，相等选定长；含 NUL 时用变长。用户可以强制定长或变长。实际宽度在 BXYZStorage 中记录，读取不依赖写入策略。

例如 Properties=species:S:1:pos:R:3:label:S:1 与 BXYZStorage=Z8:R32:S6 表示每原子 1+12+6=19 字节。全部字段定长时 BXYZPayloadBytes=N×记录字节数；包含变长字段时以 BXYZPayloadBytes 定位下一帧。未知标记、宽度不合法、字段类型不匹配、非法原子序数、无效 UTF-8、填充中夹杂非零字节和负载长度不符均报错。

## v2 兼容参考

## v2 原子记录与自适应位宽

v2 保留两行文本头，使用 `BXYZVersion=2`，并增加 `BXYZStorage`。负载改为按原子排列：每个原子内部按 Properties 顺序存储全部字段，每个字段内部按分量顺序存储，随后必须有一个额外的 LF 字节，包括最后一个原子。BXYZPayloadBytes 包含这些 LF。下一帧直接接在负载之后。N=0 时没有原子记录或额外 LF。

```text
2
Properties=species:S:1:pos:R:3 BXYZVersion=2 BXYZPayloadBytes=36 BXYZStorage=S:R32
[原子 0 的 species 和 pos 二进制数据]\n
[原子 1 的 species 和 pos 二进制数据]\n
```

上例假定两个原子都是 H：每个原子 5 字节元素字符串、12 字节位置、1 字节 LF，共 18 字节。方括号和 `\n` 为示意。数据内容自身也可能含 LF，因此必须按字段类型和字符串长度读取数据，再校验末尾 LF，不得用换行搜索分割记录。

BXYZStorage 是以冒号连接的标记，数量及顺序与 Properties 的属性严格对应，每个属性一个标记，不随原子或分量变化。Properties 本身的 R/I/L/S 类型保持不变。

| Properties 类型 | 存储标记 | 小端标量编码 |
| --- | --- | --- |
| R | R32、R64 | IEEE 754 binary32 或 binary64，4 或 8 字节 |
| I | I32、I64 | 有符号二进制补码 int32 或 int64，4 或 8 字节 |
| L | L8 | uint8，仅 0 或 1 |
| S | S | uint32 UTF-8 字节长度，随后紧跟 UTF-8 内容 |

位宽逐帧逐属性选择。写入器默认容差为 rtol=1e-7、atol=0：全部有限值经过 float32 再转回 float64 后满足 `abs(restored-original) <= atol + rtol*abs(original)` 且没有溢出时写 R32，否则写 R64。容差可配置，两者为零时要求位模式完全一致。NaN 属性保守写 R64，正负零和无穷的符号保持。整数始终无损：全部在 [-2^31, 2^31-1] 时写 I32，否则写 I64。空数值属性选择 32 位。容差是写入策略，不影响文件解码，实际类型由 BXYZStorage 唯一决定。读取数值数组统一为 float64/int64，文本头中的晶胞和其他元数据不参与位宽选择。

v2 沿用下文的字符编码、元数据和 ASE 映射约定。读取器额外拒绝缺失或不匹配的 BXYZStorage、缺失记录 LF 以及多余负载。v2 写入器采用此布局；当前写入器统一生成 v3，旧文件字节不做修改。

## v1 兼容参考

文件是零个或多个连续帧。空文件表示零帧轨迹。帧间没有填充、额外换行或索引。写入使用 LF，读取接受 CRLF 文本头；二进制区始终按字节数处理。

## 帧头

第一行是非负十进制原子数 N。第二行是 UTF-8 extxyz 键值信息，必须有 `Properties`、`BXYZVersion=1` 和非负十进制整数 `BXYZPayloadBytes`。第二行换行之后立即开始负载。负载长度不包括文本头或换行。

```text
2
Lattice="3.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 3.0" Properties=species:S:1:pos:R:3 pbc="T T T" BXYZVersion=1 BXYZPayloadBytes=58
```

上述例子若包含两个 H 原子，则 species 数据占 10 字节，pos 数据占 48 字节，负载共 58 字节。负载可以包含任意字节，包括 LF、CR 和 NUL。读取器必须消费准确的 58 字节后才读取下一帧原子数。

`Lattice` 遵循 extxyz 的矩阵排列方式：ASE 晶胞矩阵的转置，按 Fortran 顺序序列化。写入器总是输出晶胞和 PBC。缺少 PBC 时，有 Lattice 默认全周期，无 Lattice 默认非周期。其他元数据使用 ASE extxyz 约定，包括 `_JSON` 值；stress 和 virial 遵循 extxyz 特殊 3×3 矩阵约定。

## 属性与负载

`Properties` 为 `name:type:count` 三元组串，三元组之间同样用冒号连接。属性名匹配 `[A-Za-z_][A-Za-z_0-9]*`，count 为正整数。属性名与映射到 ASE 后的名字都不得重复。必须有 `pos:R:3`，且至少有 `species:S:1` 或 `Z:I:1`；两者同时存在时元素必须一致。

| 类型 | 标量编码 |
| --- | --- |
| R | IEEE 754 binary64，8 字节，小端 |
| I | 有符号二进制补码 int64，8 字节，小端 |
| L | uint8，1 字节，只允许 0 或 1 |
| S | uint32 小端 UTF-8 字节长度，随后紧跟该数量的 UTF-8 字节，无 NUL 终止符 |

每个属性的全部 N×count 个标量连续存储，再存下一个属性。属性内部顺序为 atom 0 的全部分量、atom 1 的全部分量，以此类推。无对齐填充。空字符串长度为 0，仍占 4 字节长度前缀。N=0 时每个属性占 0 字节，负载长度必须为 0。

例如 `species:S:1:pos:R:3:forces:R:3` 的负载依次为所有元素字符串、所有位置、所有力，不是逐原子交错布局。

ASE 名称映射为 `species ↔ symbols`、`pos ↔ positions`、`Z ↔ numbers`、`charge ↔ charges`。单分量属性读取为 `(N,)`，其余为 `(N,count)`。物种字符串必须能被 ASE 解释为元素；包括 ASE 的占位元素 X。

## 验证与版本

读取器拒绝缺失或未知版本、重复控制字段、负数或非整数负载长度、无效属性定义、截断字符串/数组、无效 UTF-8、逻辑字节超过 1，以及解码后剩余负载。读取器从二进制流操作文件，不允许文本模式处理负载。

v1 不支持压缩、可选数值宽度、索引或额外二进制块头。未来改变布局须使用新版本；不得将 v1 的 R 类型重新解释为 float32。

## GPUMD generic property-table extension

The local RawFrame API permits Properties without species or pos. The first line remains the number of rows (selected atoms); all properties have that row count. Physical v4/v5 encodings, byte counts, filters and checksum rules are unchanged. This extends the accepted schema, not the wire layout. Old structure-only readers reject these tables. Complete-structure Frame APIs still require pos:R:3 and species:S:1 or Z:I:1.

RawFrame.count, RawFrame.properties and RawFrame.info_extxyz represent the row count, typed arrays and metadata. Reader::next_raw/read_raw and Writer::write(RawFrame) handle generic tables; verify validates them too. Metadata and array names are independently unique, allowing total virial metadata alongside a virial array. Lattice and pbc are optional raw metadata; codec control fields remain reserved. Unknown coarse-grained species labels use S/S<W> strings rather than fabricated atomic numbers. Zero-row frames retain a nonempty property schema.

The independent Python interface is tools/binary_xyz/raw.py in the containing GPUMD repository. It yields arrays and metadata separately and does not invent missing structural fields. SHA-256 uses the OpenSSL streaming SHA256 interface for compatibility with the older compute-node OpenSSL ABI.
