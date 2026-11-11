# 工创赛智能搬运视觉单元通信协议 V1.0.0

修改日志

| 日期 | 版本 | 修订记录 |
|---|---|---|
| 2026.11.11 | V1.0.0 | 首次发布。帧格式对齐 RoboMaster 2027 通信协议 V1.0.0 第 1.1 节 |

前言

本协议定义「视觉单元（Jetson Orin）↔ 机器人主控」之间的串口通信，适用于 2027 年工程创新大赛智能搬运赛项。
帧格式、帧头定义、命令码 ID 一览的编排方式与校验算法均参照《RoboMaster 2027 机甲大师高校系列赛通信协议 V1.0.0》；
命令码与数据段内容为本赛项自行定义。

目录

1 串口协议
  1.1 串口协议格式
  1.2 命令码 ID 一览
  1.3 数据段详细定义
  1.4 字节级样例
  1.5 时序与容错
附录一：CRC 校验代码示例
附录二：坐标与角度约定
附录三：待冻结事项

# 1 串口协议

## 1.1 串口协议格式

通信方式为串口的部分，性质如下：8 位数据位，1 位停止位，无硬件流控，无校验位。详细信息见以下表格：

表 1-1 波特率

| 物理接口 | 波特率 |
|---|---|
| 视觉单元←→机器人主控 | 115200 |

表 1-2 通信协议格式

| frame_header | cmd_id | data | frame_tail |
|---|---|---|---|
| 5-byte | 2-byte | n-byte | 2-byte，CRC16，整包校验 |

表 1-3 frame_header 格式

| SOF | data_length | seq | CRC8 |
|---|---|---|---|
| 1-byte | 2-byte | 1-byte | 1-byte |

表 1-4 帧头详细定义

| 域 | 偏移位置 | 大小（字节） | 详细描述 |
|---|---|---|---|
| SOF | 0 | 1 | 数据帧起始字节，固定值为 0xA5 |
| data_length | 1 | 2 | 数据帧中 data 的长度加 2（含 cmd_id 的 2 字节），小端 |
| seq | 3 | 1 | 包序号 |
| CRC8 | 4 | 1 | 帧头 CRC8 校验，覆盖偏移 0~3 共 4 字节 |

说明：

- `data_length` 的取值范围为 2~1026。接收方若读到该范围之外的值，判定为坏帧。
- `frame_tail` 为 CRC16，覆盖范围为 `frame_header`（含 CRC8）+ `cmd_id` + `data`，即**除 `frame_tail` 自身外的全部字节**，也就是表 1-2 所称「整包校验」。
- 一帧的总字节数 = 5 + `data_length` + 2。
- 多字节字段（`data_length`、`cmd_id`、CRC16 以及所有 data 内的多字节量）一律为**小端**。
- 接收方必须实现流式解析，抗粘包与断包；帧头或 data 未收全时保留缓冲等待下一批数据，不得报错丢弃。

## 1.2 命令码 ID 一览

表 1-5 命令码 ID 一览

| 命令码 | 数据段长度 | 说明 | 发送方/接收方 |
|---|---|---|---|
| 0x0001 | 0 | 心跳，固定以 2Hz 频率发送 | 视觉单元→机器人主控 |
| 0x0002 | 12 | 任务计划，扫码成功后触发发送，每场比赛发送一次 | 视觉单元→机器人主控 |
| 0x0003 | 6 | 导航目标，触发发送，100ms 未收到 ACK 则重发 | 视觉单元→机器人主控 |
| 0x0004 | 6 | 位姿修正量，触发发送，100ms 未收到 ACK 则重发 | 视觉单元→机器人主控 |
| 0x0005 | 5 | 放置请求，触发发送，100ms 未收到 ACK 则重发 | 视觉单元→机器人主控 |
| 0x0006 | 5 | 取料请求，预留，当前不发送 | 视觉单元→机器人主控 |
| 0x0007 | 1 | 确认应答，收到任意帧后回送 | 双向 |
| 0x0201 | 8 | 实时位姿，固定以 10Hz 频率发送 | 机器人主控→视觉单元 |
| 0x0202 | 1 | 机器人状态，固定以 1Hz 频率发送 | 机器人主控→视觉单元 |
| 0x0203 | 1 | 单动作完成事件，动作结束后触发发送 | 机器人主控→视觉单元 |

命令码分段约定：`0x00xx` 为视觉单元发往机器人主控，`0x02xx` 为机器人主控发往视觉单元。
`0x01xx` 与 `0x03xx` 及更大段位保留。

## 1.3 数据段详细定义

### 表 1-6 0x0001

心跳无数据段。机器人主控收到后回送 `0x0007`。

```
typedef __packed struct
{
  /* 无数据段 */
}heartbeat_t;
```

### 表 1-7 0x0002

| 字节偏移量 | 大小 | 说明 |
|---|---|---|
| 0 | 3 | 第一批物料颜色，每字节 1~6，依次为三个物料的颜色编号 |
| 3 | 3 | 第一批放置位置，每字节 1~3，依次为三个物料的环编号 |
| 6 | 3 | 第二批物料颜色，每字节 1~6 |
| 9 | 3 | 第二批放置位置，每字节 1~3 |

```
typedef __packed struct
{
  uint8_t batch1_color[3];
  uint8_t batch1_ring[3];
  uint8_t batch2_color[3];
  uint8_t batch2_ring[3];
}task_plan_t;
```

数据段按批次分组，与二维码 4 组数字的字面顺序一一对应。例如二维码内容 `452+321+254+312` 解析为
第一批颜色 4,5,2、第一批位置 3,2,1、第二批颜色 2,5,4、第二批位置 3,1,2。

### 表 1-8 0x0003

| 字节偏移量 | 大小 | 说明 |
|---|---|---|
| 0 | 2 | 目标点 x 坐标，单位：mm，有符号 |
| 2 | 2 | 目标点 y 坐标，单位：mm，有符号 |
| 4 | 2 | 目标航向，单位：0.1°，有符号，取值范围 −1800~1800，当前恒发 0 |

```
typedef __packed struct
{
  int16_t target_x;
  int16_t target_y;
  int16_t target_heading;
}navigate_t;
```

坐标原点为扫码区角点，x 轴向右、y 轴向前为正，与 `config/field.example.yaml` 中的场地图一致。
目标点取场地图中与放置位置首位编号相同的数字环圆心；无匹配环时发送原点 `(0, 0)`。

### 表 1-9 0x0004

| 字节偏移量 | 大小 | 说明 |
|---|---|---|
| 0 | 2 | x 方向修正量，单位：mm，有符号 |
| 2 | 2 | y 方向修正量，单位：mm，有符号 |
| 4 | 2 | 航向修正量，单位：0.1°，有符号 |

```
typedef __packed struct
{
  int16_t delta_x;
  int16_t delta_y;
  int16_t delta_heading;
}calibrate_t;
```

本数据段为**修正量**而非绝对坐标：机器人主控应将三个值**叠加**到自身推算位姿上，而非直接覆盖。
修正量由视觉单元按下式计算：

```
delta = 场地图理论环心 − 视觉观测换算到场地坐标系的环心
```

其中视觉观测的换算链路为「像素坐标 →(相机安装外参)→ 车体毫米 →(机器人主控上报的实时位姿)→ 场地毫米」。
因此双方坐标系定义必须一致，详见附录二。

### 表 1-10 0x0005

| 字节偏移量 | 大小 | 说明 |
|---|---|---|
| 0 | 1 | 物料颜色编号，1~6 |
| 1 | 2 | 精细放置 x 偏移，单位：mm，有符号，当前恒发 0 |
| 3 | 2 | 精细放置 y 偏移，单位：mm，有符号，当前恒发 0 |

```
typedef __packed struct
{
  uint8_t material_color;
  int16_t fine_x;
  int16_t fine_y;
}place_t;
```

放置目标环编号已包含在 `0x0002` 的位置字节中，因此本帧不重复携带环编号。
`fine_x` / `fine_y` 为逐件精细放置的视觉微调量，当前版本恒发 0。

### 表 1-11 0x0006

| 字节偏移量 | 大小 | 说明 |
|---|---|---|
| 0 | 1 | 目标物料颜色编号，1~6 |
| 1 | 2 | 相对偏移 x，单位：mm，有符号 |
| 3 | 2 | 相对偏移 y，单位：mm，有符号 |

```
typedef __packed struct
{
  uint8_t target_color;
  int16_t offset_x;
  int16_t offset_y;
}pick_t;
```

本命令为预留命令，当前版本状态机不发送，机器人主控可选择暂不实现。

### 表 1-12 0x0007

| 字节偏移量 | 大小 | 说明 |
|---|---|---|
| 0 | 1 | 被确认帧的 seq 值 |

```
typedef __packed struct
{
  uint8_t ack_seq;
}ack_t;
```

收到任意命令帧后均应回送本帧。发送方在 100ms 内未收到对应 `ack_seq` 的应答则重发，
连续 3 次失败后进入安全停止。心跳帧不要求应答方回送，也不占用发送方的待确认槽位。

### 表 1-13 0x0201

| 字节偏移量 | 大小 | 说明 |
|---|---|---|
| 0 | 2 | 当前位置 x 坐标，单位：mm，有符号 |
| 2 | 2 | 当前位置 y 坐标，单位：mm，有符号 |
| 4 | 2 | 当前航向，单位：0.1°，有符号，取值范围 −1800~1800 |
| 6 | 2 | 累计里程，单位：m，无符号，超出 65535 后回绕 |

```
typedef __packed struct
{
  int16_t current_x;
  int16_t current_y;
  int16_t current_heading;
  uint16_t odometry;
}robot_pose_t;
```

本帧的 seq 恒为 0。

### 表 1-14 0x0202

| 字节偏移量 | 大小 | 说明 |
|---|---|---|
| 0 | 1 | 机器人状态编号，0 为空闲、1 为执行中、2 为安全停止 |

```
typedef __packed struct
{
  uint8_t robot_state;
}robot_state_t;
```

视觉单元收到后仅记录，状态机推进不依赖本帧。

### 表 1-15 0x0203

| 字节偏移量 | 大小 | 说明 |
|---|---|---|
| 0 | 1 | 完成的任务编号 |

```
typedef __packed struct
{
  uint8_t task_id;
}action_done_t;
```

每完成一个导航或放置动作发送一次，视觉单元以此推进状态机。本帧的 seq 恒为 0。

## 1.4 字节级样例

以下样例全部由视觉单元实现代码 `encode()` 实际运行输出，可直接用作收发单测的黄金样例。
复现命令（仓库根目录）：

```bash
python -c "from src.core.protocol import Frame, encode, CMD_TASK_PLAN; print(encode(Frame(1, CMD_TASK_PLAN, bytes((4,5,2,3,2,1,2,5,4,3,1,2)))).hex(' '))"
```

命令码 `0x0002`，seq=1，二维码 `452+321+254+312`（data 12 字节，`data_length` = 14 = `0x000E`，总长 21 字节）：

```
a5 0e 00 01 69 | 02 00 | 04 05 02 03 02 01 02 05 04 03 01 02 | 22 94
```

命令码 `0x0003`，seq=2，x=1200mm，y=300mm，航向 45.0°（450）：

```
a5 08 00 02 5a | 03 00 | b0 04 2c 01 c2 01 | f0 30
```

命令码 `0x0003`，seq=8，导航目标取 3 号环心 (1200, 900)，航向恒发 0：

```
a5 08 00 08 24 | 03 00 | b0 04 84 03 00 00 | 42 2e
```

命令码 `0x0004`，seq=3，修正量 x=−12mm，y=45mm，航向 8.3°（83）：

```
a5 08 00 03 04 | 04 00 | f4 ff 2d 00 53 00 | a7 df
```

命令码 `0x0005`，seq=4，物料颜色 1，精细偏移 x=−15mm，y=30mm：

```
a5 07 00 04 d8 | 05 00 | 01 f1 ff 1e 00 | a3 84
```

命令码 `0x0006`，seq=5，目标颜色 2，偏移 10mm，−20mm：

```
a5 07 00 05 86 | 06 00 | 02 0a 00 ec ff | 75 43
```

命令码 `0x0001`，seq=6，无数据段（总长 9 字节）：

```
a5 02 00 06 51 | 01 00 | 99 e5
```

命令码 `0x0007`，确认 seq=2 的帧：

```
a5 03 00 01 79 | 07 00 | 02 | e4 39
```

命令码 `0x0201`，seq=0，x=1520mm，y=300mm，航向 −123.4°（−1234），里程 1m：

```
a5 0a 00 00 a9 | 01 02 | f0 05 2c 01 2e fb 01 00 | 3e 7c
```

命令码 `0x0203`，seq=0，任务编号 7：

```
a5 03 00 00 27 | 03 02 | 07 | 88 4e
```

命令码 `0x0202`，seq=0，状态编号 3：

```
a5 03 00 00 27 | 02 02 | 03 | 70 52
```

命令码 `0x0003`，seq=7，边界值 x=32767、y=−32768、航向 −180.0°（−18000）：

```
a5 08 00 07 65 | 03 00 | ff 7f 00 80 b0 b9 | 69 eb
```

## 1.5 时序与容错

表 1-16 时序参数

| 参数 | 值 | 说明 |
|---|---|---|
| 心跳周期 | 500ms | 视觉单元以 2Hz 发送 `0x0001` |
| 断链超时 | 2s | 任一方超过 2s 未收到任何来帧则进入安全停止 |
| 应答超时 | 100ms | 未收到 `0x0007` 则重发 |
| 应答重试上限 | 3 次 | 连续 3 次失败进入安全停止 |
| 位姿上报频率 | 10Hz | 机器人主控以 10Hz 发送 `0x0201` |
| 动作超时 | 8s | 导航或放置动作超过 8s 未收到 `0x0203` 则重试，重试上限 2 次 |
| 校准超时 | 6s | 校准窗口超过 6s 未成功则跳过校准继续比赛 |

容错规则：

- 校验失败（`data_length` 越界、CRC8 不符、CRC16 不符）的帧整帧丢弃并计数。
- 丢弃坏帧时**只丢弃 1 字节**，随后从下一个 0xA5 重新查找帧头，不得清空整个接收缓冲。
  这样粘包场景下一帧损坏不会连带丢弃其后的好帧。
- data 内部出现 0xA5 不会被误判为帧头，由 CRC8 与 CRC16 双重校验排除。
- 控制帧（`0x0002`~`0x0006`）串行化发送：上一条控制帧的应答未解决（含重发预算未耗尽）之前，
  不发送下一条控制帧。心跳不受此限制。
- 数据段长度与表 1-5 不符的帧视为实现不匹配，丢弃并记录，不得使程序崩溃。

# 附录一：CRC 校验代码示例

```c
//crc8 generator polynomial :G(x)=x8+x5+x4+1
const unsigned char CRC8_INIT = 0xff;
const unsigned char CRC8_TAB[256] =
{
0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83, 0xc2, 0x9c, 0x7e, 0x20, 0xa3, 0xfd, 0x1f,
0x41,
0x9d, 0xc3, 0x21, 0x7f, 0xfc, 0xa2, 0x40, 0x1e, 0x5f, 0x01, 0xe3, 0xbd, 0x3e, 0x60, 0x82,
0xdc, 0x23, 0x7d, 0x9f, 0xc1, 0x42, 0x1c, 0xfe, 0xa0, 0xe1, 0xbf, 0x5d, 0x03, 0x80, 0xde,
0x3c, 0x62, 0xbe, 0xe0, 0x02, 0x5c, 0xdf, 0x81, 0x63, 0x3d, 0x7c, 0x22, 0xc0, 0x9e, 0x1d,
0x43, 0xa1, 0xff, 0x46, 0x18, 0xfa, 0xa4, 0x27, 0x79, 0x9b, 0xc5, 0x84, 0xda, 0x38, 0x66, 0xe5,
0xbb, 0x59, 0x07, 0xdb, 0x85, 0x67, 0x39, 0xba, 0xe4, 0x06, 0x58, 0x19, 0x47, 0xa5, 0xfb,
0x78, 0x26, 0xc4, 0x9a, 0x65, 0x3b, 0xd9, 0x87, 0x04, 0x5a, 0xb8, 0xe6, 0xa7, 0xf9, 0x1b,
0x45, 0xc6, 0x98, 0x7a, 0x24, 0xf8, 0xa6, 0x44, 0x1a, 0x99, 0xc7, 0x25, 0x7b, 0x3a, 0x64,
0x86, 0xd8, 0x5b, 0x05, 0xe7, 0xb9,
0x8c, 0xd2, 0x30, 0x6e, 0xed, 0xb3, 0x51, 0x0f, 0x4e, 0x10, 0xf2, 0xac, 0x2f, 0x71, 0x93,
0xcd, 0x11, 0x4f, 0xad, 0xf3, 0x70, 0x2e, 0xcc, 0x92, 0xd3, 0x8d, 0x6f, 0x31, 0xb2, 0xec,
0x0e, 0x50, 0xaf, 0xf1, 0x13, 0x4d, 0xce, 0x90, 0x72, 0x2c, 0x6d, 0x33, 0xd1, 0x8f, 0x0c, 0x52,
0xb0, 0xee, 0x32, 0x6c, 0x8e, 0xd0, 0x53, 0x0d, 0xef, 0xb1, 0xf0, 0xae, 0x4c, 0x12, 0x91,
0xcf, 0x2d, 0x73, 0xca, 0x94, 0x76, 0x28, 0xab, 0xf5, 0x17, 0x49, 0x08, 0x56, 0xb4, 0xea,
0x69, 0x37, 0xd5, 0x8b, 0x57, 0x09, 0xeb, 0xb5, 0x36, 0x68, 0x8a, 0xd4, 0x95, 0xcb, 0x29,
0x77, 0xf4, 0xaa, 0x48, 0x16, 0xe9, 0xb7, 0x55, 0x0b, 0x88, 0xd6, 0x34, 0x6a, 0x2b, 0x75,
0x97, 0xc9, 0x4a, 0x14, 0xf6, 0xa8,
0x74, 0x2a, 0xc8, 0x96, 0x15, 0x4b, 0xa9, 0xf7, 0xb6, 0xe8, 0x0a, 0x54, 0xd7, 0x89, 0x6b,
0x35,
};
unsigned char Get_CRC8_Check_Sum(unsigned char *pchMessage,unsigned int
dwLength,unsigned char ucCRC8)
{
unsigned char ucIndex;
while (dwLength --)
{
ucIndex = ucCRC8^(*pchMessage++);
ucCRC8 = CRC8_TAB[ucIndex];
}
return(ucCRC8);
}
/*
** Descriptions : CRC8 Verify function
** Input : Data to Verify,Stream length = Data + checksum
** Output : True or False (CRC Verify Result)
*/
unsigned int Verify_CRC8_Check_Sum(unsigned char *pchMessage, unsigned int
dwLength)
{
unsigned char ucExpected = 0;
if ((pchMessage == 0) || (dwLength <= 2)) return 0;
ucExpected = Get_CRC8_Check_Sum (pchMessage, dwLength -1, CRC8_INIT);
return ( ucExpected == pchMessage[dwLength -1] );
}
/*
** Descriptions : append CRC8 to the end of data
** Input : Data to CRC and append,Stream length = Data + checksum
** Output : True or False (CRC Verify Result)
*/
void Append_CRC8_Check_Sum(unsigned char *pchMessage, unsigned int dwLength)
{
unsigned char ucCRC = 0;
if ((pchMessage == 0) || (dwLength <= 2)) return;
ucCRC = Get_CRC8_Check_Sum ( (unsigned char *)pchMessage, dwLength -1, CRC8_INIT);
pchMessage[dwLength -1] = ucCRC;
}
uint16_t CRC_INIT = 0xffff;
const uint16_t wCRC_Table[256] =
{
0x0000, 0x1189, 0x2312, 0x329b, 0x4624, 0x57ad, 0x6536, 0x74bf,
0x8c48, 0x9dc1, 0xaf5a, 0xbed3, 0xca6c, 0xdbe5, 0xe97e, 0xf8f7,
0x1081, 0x0108, 0x3393, 0x221a, 0x56a5, 0x472c, 0x75b7, 0x643e,
0x9cc9, 0x8d40, 0xbfdb, 0xae52, 0xdaed, 0xcb64, 0xf9ff, 0xe876,
0x2102, 0x308b, 0x0210, 0x1399, 0x6726, 0x76af, 0x4434, 0x55bd,
0xad4a, 0xbcc3, 0x8e58, 0x9fd1, 0xeb6e, 0xfae7, 0xc87c, 0xd9f5,
0x3183, 0x200a, 0x1291, 0x0318, 0x77a7, 0x662e, 0x54b5, 0x453c,
0xbdcb, 0xac42, 0x9ed9, 0x8f50, 0xfbef, 0xea66, 0xd8fd, 0xc974,
0x4204, 0x538d, 0x6116, 0x709f, 0x0420, 0x15a9, 0x2732, 0x36bb,
0xce4c, 0xdfc5, 0xed5e, 0xfcd7, 0x8868, 0x99e1, 0xab7a, 0xbaf3,
0x5285, 0x430c, 0x7197, 0x601e, 0x14a1, 0x0528, 0x37b3, 0x263a,
0xdecd, 0xcf44, 0xfddf, 0xec56, 0x98e9, 0x8960, 0xbbfb, 0xaa72,
0x6306, 0x728f, 0x4014, 0x519d, 0x2522, 0x34ab, 0x0630, 0x17b9,
0xef4e, 0xfec7, 0xcc5c, 0xddd5, 0xa96a, 0xb8e3, 0x8a78, 0x9bf1,
0x7387, 0x620e, 0x5095, 0x411c, 0x35a3, 0x242a, 0x16b1, 0x0738,
0xffcf, 0xee46, 0xdcdd, 0xcd54, 0xb9eb, 0xa862, 0x9af9, 0x8b70,
0x8408, 0x9581, 0xa71a, 0xb693, 0xc22c, 0xd3a5, 0xe13e, 0xf0b7,
0x0840, 0x19c9, 0x2b52, 0x3adb, 0x4e64, 0x5fed, 0x6d76, 0x7cff,
0x9489, 0x8500, 0xb79b, 0xa612, 0xd2ad, 0xc324, 0xf1bf, 0xe036,
0x18c1, 0x0948, 0x3bd3, 0x2a5a, 0x5ee5, 0x4f6c, 0x7df7, 0x6c7e,
0xa50a, 0xb483, 0x8618, 0x9791, 0xe32e, 0xf2a7, 0xc03c, 0xd1b5,
0x2942, 0x38cb, 0x0a50, 0x1bd9, 0x6f66, 0x7eef, 0x4c74, 0x5dfd,
0xb58b, 0xa402, 0x9699, 0x8710, 0xf3af, 0xe226, 0xd0bd, 0xc134,
0x39c3, 0x284a, 0x1ad1, 0x0b58, 0x7fe7, 0x6e6e, 0x5cf5, 0x4d7c,
0xc60c, 0xd785, 0xe51e, 0xf497, 0x8028, 0x91a1, 0xa33a, 0xb2b3,
0x4a44, 0x5bcd, 0x6956, 0x78df, 0x0c60, 0x1de9, 0x2f72, 0x3efb,
0xd68d, 0xc704, 0xf59f, 0xe416, 0x90a9, 0x8120, 0xb3bb, 0xa232,
0x5ac5, 0x4b4c, 0x79d7, 0x685e, 0x1ce1, 0x0d68, 0x3ff3, 0x2e7a,
0xe70e, 0xf687, 0xc41c, 0xd595, 0xa12a, 0xb0a3, 0x8238, 0x93b1,
0x6b46, 0x7acf, 0x4854, 0x59dd, 0x2d62, 0x3ceb, 0x0e70, 0x1ff9,
0xf78f, 0xe606, 0xd49d, 0xc514, 0xb1ab, 0xa022, 0x92b9, 0x8330,
0x7bc7, 0x6a4e, 0x58d5, 0x495c, 0x3de3, 0x2c6a, 0x1ef1, 0x0f78
};
/*
** Descriptions : CRC16 checksum function
** Input : Data to check,Stream length, initialized checksum
** Output : CRC checksum
*/
uint16_t Get_CRC16_Check_Sum(uint8_t *pchMessage,uint32_t dwLength,uint16_t wCRC)
{
Uint8_t chData;
if (pchMessage == NULL)
{
return 0xFFFF;
}
while(dwLength --)
{
chData = *pchMessage++;
(wCRC) = ((uint16_t)(wCRC) >> 8) ^ wCRC_Table[((uint16_t)(wCRC) ^ (uint16_t)(chData)) &
0x00ff];
}
return wCRC;
}
/*
** Descriptions : CRC16 Verify function
** Input : Data to Verify ,Stream length = Data + checksum
** Output : True or False (CRC Verify Result)
*/
uint32_t Verify_CRC16_Check_Sum(uint8_t *pchMessage, uint32_t dwLength)
{
uint16_t wExpected = 0;
if ((pchMessage == NULL) || (dwLength <= 2))
{
return __FALSE;
}
wExpected = Get_CRC16_Check_Sum ( pchMessage, dwLength - 2, CRC_INIT);
return ((wExpected & 0xff) == pchMessage[dwLength - 2] && ((wExpected >> 8) & 0xff) ==
pchMessage[dwLength - 1]);
}
/*
** Descriptions : append CRC16 to the end of data
** Input : Data to CRC and append,Stream length = Data + checksum
** Output : True or False (CRC Verify Result)
*/
void Append_CRC16_Check_Sum(uint8_t * pchMessage,uint32_t dwLength)
{
uint16_t wCRC = 0;
if ((pchMessage == NULL) || (dwLength <= 2))
{
return;
}
wCRC = Get_CRC16_Check_Sum ( (U8 *)pchMessage, dwLength -2, CRC_INIT );
pchMessage[dwLength -2] = (U8)(wCRC & 0x00ff);
pchMessage[dwLength -1] = (U8)((wCRC >> 8)& 0x00ff);
}
```

视觉单元侧的 Python 等价实现位于 `src/core/crc.py`，参数与上表完全一致：

- CRC8：生成多项式 G(x)=x8+x5+x4+1（0x31 的反射形式 0x8C，LSB-first），初值 0xFF，无结果异或。
- CRC16：生成多项式 0x1021 的反射形式 0x8408（LSB-first），初值 0xFFFF，无结果异或。

上表 256 项 CRC8 与 CRC16 表在 `src/core/crc.py` 中由算法生成而非手抄，
`tests/test_crc.py` 用本附录表中首行 16 个值与末值 0x35 逐项固化对账。

# 附录二：坐标与角度约定

一、坐标系

- 原点为扫码区角点，x 轴向右为正，y 轴向前为正，单位 mm。
- 视觉单元与机器人主控必须使用同一套坐标系。若定义不一致，校准修正量会整体错位。
- 场地尺寸与环节坐标见 `config/field.example.yaml`，双方应各自持有一份相同数据。

二、角度

- 传输单位统一为 0.1°，即 1.0° 发送为 10。
- 有效取值范围为 −1800~1800，对应 −180.0°~180.0°。
- 发送方应先将角度归一到该区间再打包：

```
value = ((raw + 1800) % 3600) - 1800
```

三、量程与饱和

- 坐标字段为 int16，单方向表示范围 −32768~32767 mm，约 ±32.7 m。
- 超出量程时按**饱和钳位**处理，不回绕。接收方无需处理回绕歧义。

四、校准依赖

- 视觉单元使用机器人主控最近上报的 `0x0201` 位姿做坐标换算，10Hz 上报下延迟不超过 100ms。
- 机器人主控在校准窗口内应保持静止，否则视觉采样得到的修正量会包含运动误差。

# 附录三：待冻结事项

联调前需与电控组逐项确认并签署冻结：

1. 帧格式与校验算法（表 1-2、表 1-3、表 1-4、附录一）
2. 命令码 ID 一览（表 1-5）
3. 各命令数据段定义与长度（表 1-6 至表 1-15）
4. 坐标系原点、坐标轴方向与单位（附录二）
5. 时序参数：心跳 500ms、断链 2s、应答 100ms × 3 次（表 1-16）
6. 安全停止的触发条件与恢复责任方
7. 预留命令 `0x0006` 的实现取舍
