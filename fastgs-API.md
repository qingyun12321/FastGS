# FastGS API 文档

适用默认项目：`fastgs`

## 1. 服务地址

### 1.1 Task-Manager

- 默认地址：`http://36.133.236.108:8090`
- 恢复任务：`POST /api/task/recover`
- 暂停任务：`POST /api/task/pause`

请求体（recover / pause 一致）：

```json
{
  "project": "fastgs"
}
```

### 1.2 Runtime API

运行时服务基地址使用 `recover` 返回的 `service_url`。以下接口均以该地址为前缀：

- `GET /health`
- `POST /create_session`
- `POST /run_with_files`
- `GET /queue_status`
- `GET /request_status`
- `GET /logs`
- `POST /cleanup`

后端默认端口（`http_api.py`）：`8000`。

### 1.3 URL 参数（Web 控制台）

- `task_manager`：覆盖 task-manager 地址
- `project`：覆盖任务项目名

示例：

```text
http://<frontend-host>/?task_manager=http://10.0.0.8:8090&project=fastgs-prod
```

## 2. 调用顺序

1. `POST {task_manager}/api/task/recover`
2. `GET {service_url}/health`（轮询直到可用）
3. `POST {service_url}/create_session`
4. `POST {service_url}/run_with_files`
5. 轮询任务状态
   - `GET {service_url}/queue_status?request_id=...`
   - `GET {service_url}/request_status?request_id=...`
   - `GET {service_url}/logs?session_id=...&max_lines=400`
6. `POST {task_manager}/api/task/pause`
7. `POST {service_url}/cleanup`（可选，页面关闭时可调用）

## 3. 接口定义

### 3.1 `POST /api/task/recover`

请求体：

```json
{
  "project": "fastgs"
}
```

关键响应字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `service_url` | string | 是 | Runtime API 基地址 |
| `recovered` | boolean | 否 | 本次请求是否触发恢复 |

失败语义：

- 非 2xx：恢复失败
- 返回 `service_url` 为空：不可继续后续调用

### 3.2 `POST /api/task/pause`

请求体：

```json
{
  "project": "fastgs"
}
```

用途：释放算力任务。

### 3.3 `GET /health`

请求参数：无。

响应：

```json
{
  "status": "ok"
}
```

### 3.4 `POST /create_session`

请求体：无。

响应：

```json
{
  "session_id": "8bb5c880-1caa-4d9f-93a2-c89dbf84f278"
}
```

### 3.5 `POST /run_with_files`

请求类型：`multipart/form-data`

#### 3.5.1 表单参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `session_id` | string | 否 | 自动创建 | 会话 ID |
| `request_id` | string | 否 | 自动生成 UUID | 请求 ID |
| `iterations` | int | 否 | `30000` | 训练轮数 |
| `mult` | float | 否 | `0.5` | 渲染倍率 |
| `white_background` | bool | 否 | `false` | 白色背景 |
| `eval` | bool | 否 | `false` | 评估模式 |
| `data_device` | string | 否 | `cuda` | 数据设备 |
| `iteration` | int | 否 | `-1` | 读取迭代点 |
| `video360` | bool | 否 | `false` | 生成 360 视频 |
| `use_dataset_cams` | bool | 否 | `false` | 使用数据集相机插帧 |
| `use_test_cams` | bool | 否 | `false` | 使用测试相机插帧 |
| `interp_per_pair` | int | 否 | `3` | 相机插值帧/对 |
| `loop` | bool | 否 | `false` | 轨迹闭环 |
| `fps` | int | 否 | `30` | 视频帧率 |
| `frames` | int | 否 | `240` | 轨道渲染总帧数 |
| `ease` | bool | 否 | `true` | 轨道缓动 |
| `render_only` | bool | 否 | `false` | 仅渲染，不训练 |
| `dataset_file` | file `.zip` | 条件必填 | 无 | 训练模式必填 |
| `model_file` | file `.zip` | 条件必填 | 无 | `render_only=true` 时必填 |
| `pose_file` | file `.json` | 否 | 无 | Pose 渲染输入 |

#### 3.5.2 校验与错误

- `dataset_file` 非 `.zip`：`400`
- `model_file` 非 `.zip`：`400`
- `pose_file` 非 `.json`：`400`
- 训练模式缺少 `dataset_file`：`409`
- `render_only=true` 且缺少 `model_file`：`409`
- `request_id` 与队列中待处理/处理中任务冲突：`409`

#### 3.5.3 成功响应

```json
{
  "status": "queued",
  "request_id": "b7260a6f-b17d-40cf-88cc-6f8a59473e7a",
  "position": 1,
  "session_id": "8bb5c880-1caa-4d9f-93a2-c89dbf84f278"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | 固定为 `queued` |
| `request_id` | string | 请求 ID |
| `position` | int | 队列位置（`1` 为队首） |
| `session_id` | string | 会话 ID |

### 3.6 `GET /queue_status`

Query 参数：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `request_id` | string | 否 | 指定后返回该请求状态与位置 |

响应字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `processing` | boolean | 是否有任务在执行 |
| `pending` | int | 排队任务数量 |
| `current_request_id` | string | 当前执行请求 ID，无则空字符串 |
| `status` | string | 传 `request_id` 时返回：`pending` / `processing` / `completed` / `failed` / `unknown` |
| `position` | int | 传 `request_id` 时返回 |

`position` 语义：

- `0`：请求正在执行
- `>=1`：请求在队列中（`1` 为队首）
- `-1`：未找到或不在等待队列

### 3.7 `GET /request_status`

Query 参数：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `request_id` | string | 是 | 请求 ID |

错误码：

- `400`：`request_id` 为空
- `404`：`request_id` 不存在

响应字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `request_id` | string | 请求 ID |
| `status` | string | `pending` / `processing` / `completed` / `failed` |
| `error` | string | 失败原因 |
| `started_at` | number \| null | 开始时间（Unix 秒） |
| `finished_at` | number \| null | 结束时间（Unix 秒） |
| `result` | object | 完成后包含产物信息 |

`result` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `session_id` | string | 会话 ID |
| `oss_prefix` | string | OSS 输出目录前缀 |
| `artifacts` | object | 输出产物 |

`artifacts` 子字段（按输出可能为 `null`）：

| 字段 | 类型 | 文件 |
|---|---|---|
| `ply` | object \| null | `point_cloud.ply` |
| `video` | object \| null | `orbit_360.mp4` |
| `pose` | object \| null | `pose_*.png` |
| `bundle` | object \| null | `fastgs_output.zip` |
| `log` | object \| null | `train.log` |

每个产物对象结构：

```json
{
  "oss_key": "docker-input&output/fastgs/<session_id>/<request_id>/output/point_cloud.ply",
  "url": "https://..."
}
```

### 3.8 `GET /logs`

Query 参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `session_id` | string | 否 | 空 | 指定会话日志 |
| `max_lines` | int | 否 | `200` | 返回最后 N 行（范围 `1~2000`） |

响应：

```json
{
  "log": "...\n"
}
```

### 3.9 `POST /cleanup`

请求类型：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `session_id` | string | 是 | 需清理的会话 ID |

成功响应：

```json
{
  "status": "cleaned"
}
```

错误码：

- `400`：`session_id` 为空

## 4. 示例

### 4.1 创建会话

```bash
curl -X POST "http://127.0.0.1:8000/create_session"
```

### 4.2 直传并入队

```bash
curl -X POST "http://127.0.0.1:8000/run_with_files" \
  -F "session_id=<session_id>" \
  -F "request_id=req-001" \
  -F "iterations=30000" \
  -F "mult=0.5" \
  -F "render_only=false" \
  -F "video360=true" \
  -F "use_dataset_cams=true" \
  -F "interp_per_pair=3" \
  -F "fps=30" \
  -F "dataset_file=@/path/to/dataset.zip"
```

### 4.3 查询状态

```bash
curl "http://127.0.0.1:8000/queue_status?request_id=req-001"
curl "http://127.0.0.1:8000/request_status?request_id=req-001"
```
