# 模板截图库（T2 定位素材）

模板图是**外置素材**：现场可通过配置页「运行实况 → 重新截取模板」拉框更新，无需发版。
T1 OCR 文字锚点是默认定位方式，以下模板仅作 T1 失败时的降级（fallback）或无文字元素定位用，
**缺失时不影响 T1 正常工作**。

| 文件名 | 用途 |
|---|---|
| `login_page.png` | 登录页特征（状态判定辅助，可选） |
| `username_input.png` | 用户名输入框（USERNAME_INPUT 的 T2 降级） |
| `password_input.png` | 密码输入框（PASSWORD_INPUT 的 T2 降级） |
| `login_button.png` | 登录按钮（LOGIN_BUTTON 的 T2 降级） |
| `main_page.png` | 主页面特征（状态判定辅助，可选） |
| `vehicle_input.png` | 车辆自编号输入框（VEHICLE_INPUT 的 T2 降级） |
| `print_button.png` | 打印按钮（PRINT_BUTTON 的 T2 降级） |
| `print_success.png` | 打印成功特征（verify_success 开启时校验用，可选） |

截取要领（README §2.1）：多框进一圈周边上下文，上下文让模板天然唯一。
