# kokoni-gs API

## 1. 接入概览

`kokoni-gs` 采用异步任务模式：

1. 提交任务
2. 获取 `task_id`
3. 轮询任务状态
4. 任务完成后读取结果文件 URL

## 2. 服务地址

当前服务地址（下文中的 `base_url`）：

```text
http://36.133.236.108:8090
```

## 3. 鉴权

API 使用 **Bearer Token** 机制进行访问控制。客户端需要在 Header 中传递 `Authorization` 字段。

| Header Field | Value Format | 说明 |
|---|---|---|
| `Authorization` | `Bearer <YOUR_API_KEY>` | 请将 `<YOUR_API_KEY>` 替换为实际分配的密钥 |

## 4. 接口清单

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/v1/services/aigc/3d-generation/reconstruction` | 创建 FastGS 任务 |
| `GET` | `/api/v1/tasks/{task_id}` | 查询任务状态与结果 |

## 5. 创建任务

### 5.1 请求地址

```text
POST http://36.133.236.108:8090/api/v1/services/aigc/3d-generation/reconstruction
```

### 5.2 请求类型

`multipart/form-data`

### 5.3 请求参数

表单中包含一个 JSON part 和若干文件 part：

| 参数名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `request` | string | 是 | JSON 字符串，外层结构固定为 `model / input / parameters` |
| `dataset_file` | file `.zip` | 条件必填 | 训练模式必填 |
| `model_file` | file `.zip` | 条件必填 | `render_only=true` 时必填 |
| `pose_file` | file `.json` | 否 | Pose 渲染输入 |

### 5.3.1 上传文件要求

#### `dataset_file`

`dataset_file` 必须是一个 `.zip`，解压后目录需要满足 FastGS 可识别的数据集格式之一：

1. **COLMAP 格式**

至少包含：

```text
dataset_root/
  images/
  sparse/
    0/
      cameras.bin | cameras.txt
      images.bin  | images.txt
      points3D.bin | points3D.txt
```

2. **Blender / NeRF Synthetic 格式**

至少包含：

```text
dataset_root/
  transforms_train.json
  transforms_test.json
  ...
```

并且 `transforms_train.json` / `transforms_test.json` 中引用到的图像文件必须真实存在于压缩包内。

#### `model_file`

`model_file` 必须是 `.zip`，解压后应包含可用于 FastGS 渲染的模型输出目录内容。  
当 `render_only=true` 时必须提供。

#### `pose_file`

`pose_file` 必须是 `.json`，用于 Pose 渲染输入。  
只有在需要 Pose 渲染时才需要上传。

#### 不建议的输入

以下输入通常无法直接跑通：

- 仅包含一张图片的 `.zip`
- 仅包含 `images/`、但没有 `sparse/0/` 的 `.zip`
- 仅包含若干图片、但没有 `transforms_train.json` 的 `.zip`

如果只是为了验证链路，建议优先准备一个最小可用的 Blender/NeRF Synthetic 数据集包，或一个完整的 COLMAP 数据集包。

### 5.4 `request` 字段说明

```json
{
  "model": "kokoni-gs",
  "input": {
    "request_id": "optional-client-id"
  },
  "parameters": {
    "iterations": 30000,
    "mult": 0.5,
    "white_background": false,
    "eval": false,
    "data_device": "cuda",
    "iteration": -1,
    "video360": false,
    "use_dataset_cams": true,
    "use_test_cams": false,
    "interp_per_pair": 3,
    "loop": false,
    "fps": 30,
    "frames": 240,
    "ease": true,
    "render_only": false
  }
}
```

#### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `model` | string | 是 | 模型名称，当前使用 `kokoni-gs` |
| `input` | object | 是 | 输入参数 |
| `parameters` | object | 是 | 控制参数 |

#### `input` 字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `request_id` | string | 否 | 自动生成 | 客户端请求 ID，建议用于链路追踪 |

#### `parameters` 字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `iterations` | integer | 否 | `30000` | 训练轮数 |
| `mult` | number | 否 | `0.5` | 渲染倍率 |
| `white_background` | boolean | 否 | `false` | 白色背景 |
| `eval` | boolean | 否 | `false` | 评估模式 |
| `data_device` | string | 否 | `cuda` | 数据设备 |
| `iteration` | integer | 否 | `-1` | 读取迭代点 |
| `video360` | boolean | 否 | `false` | 生成 360 视频 |
| `use_dataset_cams` | boolean | 否 | `true` | 使用数据集相机插帧 |
| `use_test_cams` | boolean | 否 | `false` | 使用测试相机插帧 |
| `interp_per_pair` | integer | 否 | `3` | 相机插值帧/对 |
| `loop` | boolean | 否 | `false` | 轨迹闭环 |
| `fps` | integer | 否 | `30` | 视频帧率 |
| `frames` | integer | 否 | `240` | 轨道渲染总帧数 |
| `ease` | boolean | 否 | `true` | 轨道缓动 |
| `render_only` | boolean | 否 | `false` | 仅渲染，不训练 |

### 5.5 请求示例

```bash
curl --location 'http://36.133.236.108:8090/api/v1/services/aigc/3d-generation/reconstruction' \
  -H 'Authorization: Bearer <YOUR_API_KEY>' \
  -F 'request={
    "model":"kokoni-gs",
    "input":{"request_id":"req-001"},
    "parameters":{
      "iterations":30000,
      "mult":0.5,
      "white_background":false,
      "eval":false,
      "data_device":"cuda",
      "iteration":-1,
      "video360":false,
      "use_dataset_cams":true,
      "use_test_cams":false,
      "interp_per_pair":3,
      "loop":false,
      "fps":30,
      "frames":240,
      "ease":true,
      "render_only":false
    }
  }' \
  -F 'dataset_file=@/path/to/dataset.zip'
```

训练模式最小验证时：

- 只上传 `dataset_file` 即可
- 但 `dataset_file.zip` 必须符合上面的数据集目录要求

### 5.6 成功响应示例

```json
{
  "status_code": 200,
  "request_id": "req-001",
  "code": null,
  "message": "",
  "output": {
    "task_id": "4bb0f773cb7c4f6fa15f1173d46c7db7",
    "task_status": "PENDING",
    "submit_time": "2026-03-12 12:34:56.789"
  }
}
```

## 6. 查询任务状态

### 6.1 请求地址

```text
GET http://36.133.236.108:8090/api/v1/tasks/{task_id}
```

### 6.2 路径参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `task_id` | string | 是 | 创建任务接口返回的任务 ID |

### 6.3 任务状态说明

| 状态 | 说明 |
|---|---|
| `PENDING` | 任务已创建，等待可用算力 |
| `SCALING` | 平台正在准备可用算力，请继续轮询 |
| `RUNNING` | 任务正在执行 |
| `SUCCEEDED` | 任务完成，可读取结果 |
| `FAILED` | 任务失败，请查看 `message` |

### 6.4 查询响应字段

响应根级字段固定如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `status_code` | integer | 接口状态码 |
| `request_id` | string | 请求 ID |
| `code` | string \| null | 业务码 |
| `message` | string | 状态说明或错误信息 |
| `output` | object | 任务信息与结果 |

`output` 中固定包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `task_id` | string | 任务 ID |
| `task_status` | string | 任务状态 |
| `submit_time` | string | 提交时间 |
| `scheduled_time` | string \| null | 调度时间 |
| `start_time` | string \| null | 开始执行时间 |
| `end_time` | string \| null | 结束时间 |

任务成功后，`output` 中还会包含以下结果字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `session_id` | string | 会话 ID |
| `oss_prefix` | string | OSS 输出目录前缀 |
| `artifacts` | object | 输出产物集合 |

`artifacts` 可能包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `ply` | object \| null | PLY 点云结果 |
| `video` | object \| null | 360 视频结果 |
| `pose` | object \| null | Pose 渲染图 |
| `bundle` | object \| null | 打包 ZIP |
| `log` | object \| null | 训练日志 |

每个产物对象结构如下：

```json
{
  "oss_key": "docker-input&output/fastgs/<session_id>/<request_id>/output/point_cloud.ply",
  "url": "https://..."
}
```

### 6.5 查询示例

```bash
curl --location 'http://36.133.236.108:8090/api/v1/tasks/4bb0f773cb7c4f6fa15f1173d46c7db7' \
  -H 'Authorization: Bearer <YOUR_API_KEY>'
```

### 6.6 成功完成响应示例

```json
{
  "status_code": 200,
  "request_id": "req-001",
  "code": null,
  "message": "",
  "output": {
    "task_id": "4bb0f773cb7c4f6fa15f1173d46c7db7",
    "task_status": "SUCCEEDED",
    "submit_time": "2026-03-12 12:34:56.789",
    "scheduled_time": "2026-03-12 12:35:00.100",
    "start_time": "2026-03-12 12:35:01.230",
    "end_time": "2026-03-12 12:42:18.456",
    "session_id": "8bb5c880-1caa-4d9f-93a2-c89dbf84f278",
    "oss_prefix": "docker-input&output/fastgs/8bb5c880-1caa-4d9f-93a2-c89dbf84f278/req-001",
    "artifacts": {
      "ply": {"oss_key": "docker-input&output/fastgs/.../point_cloud.ply", "url": "https://example.com/point_cloud.ply"},
      "video": {"oss_key": "docker-input&output/fastgs/.../orbit_360.mp4", "url": "https://example.com/orbit_360.mp4"},
      "pose": null,
      "bundle": {"oss_key": "docker-input&output/fastgs/.../fastgs_output.zip", "url": "https://example.com/fastgs_output.zip"},
      "log": {"oss_key": "docker-input&output/fastgs/.../train.log", "url": "https://example.com/train.log"}
    }
  }
}
```

## 7. 推荐调用方式

1. 调用创建任务接口，获取 `task_id`
2. 每 2 秒轮询一次任务状态接口
3. 当 `task_status=SUCCEEDED` 时读取结果文件 URL
4. 当 `task_status=FAILED` 时提示失败原因并决定是否重试

## 8. 错误处理建议

| 场景 | 建议处理方式 |
|---|---|
| 训练模式缺少 `dataset_file` | 补充数据集 ZIP 后重试 |
| `render_only=true` 且缺少 `model_file` | 补充模型 ZIP 后重试 |
| `pose_file` 不是 `.json` | 更换正确文件格式 |
| 任务长时间处于 `SCALING` | 平台正在准备可用算力，可继续轮询 |
| 任务返回 `FAILED` | 展示 `message`，并根据业务决定是否重新提交 |

## 9. 结果文件说明

- `ply`：点云结果
- `video`：轨道或 360 视频
- `pose`：单张 Pose 渲染图
- `bundle`：结果打包 ZIP
- `log`：训练日志

建议在业务侧自行管理下载、缓存和过期策略。
