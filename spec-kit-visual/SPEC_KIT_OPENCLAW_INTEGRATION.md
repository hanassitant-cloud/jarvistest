# Spec Kit × OpenClaw / 文文整合研究方案

## 目標

把 GitHub Spec Kit 的 Spec-Driven Development（規格驅動開發）流程，整理成文文可重複使用的 AI 顧問與企業系統開發 SOP。

核心目標不是單純安裝工具，而是建立一套從「客戶需求 → 規格 → 技術計畫 → 任務 → 實作 → 驗收」的可追蹤流程。

## Spec Kit 核心流程

1. `constitution`：建立專案原則與開發規範。
2. `specify`：描述要做什麼與為什麼，不先談技術。
3. `clarify`：補足模糊需求。
4. `plan`：把規格轉成技術計畫。
5. `tasks`：拆成可執行任務。
6. `analyze / checklist`：檢查規格、計畫、任務是否一致。
7. `implement`：依任務實作。

## 對文文的價值

- 企業 AI 顧問導入時，可以先產出規格，而不是直接寫程式。
- 客戶需求可以版本化，避免口頭需求變動造成混亂。
- 技術方案與商業需求可以雙向追蹤。
- 可與 OpenClaw sub-agent / TaskFlow 概念結合：不同 agent 負責研究、規格、計畫、實作、驗收。
- 適合 Jason 未來幫企業導入 AI 系統、LINE/Telegram 助理、知識庫、客服、內部工具與報表系統。

## 建議導入方式

### Phase 1：先不安裝，只採用方法論

建立文文自己的「AI 顧問規格模板」：

- 專案背景
- 使用者角色
- 核心場景
- 功能需求
- 非功能需求
- 資料來源與權限
- 風險與不可做事項
- 驗收標準
- 90 天導入任務

### Phase 2：小專案試點

選一個低風險專案測試，例如：

- 診所 LINE AI 助理
- 企業 FAQ 知識庫
- HTML PPT 自動產生器
- 客服問答機器人

用 Spec Kit 產出 `spec.md`、`plan.md`、`tasks.md`，再交給 coding agent 實作。

### Phase 3：建立 OpenClaw Skill / Preset

長期可以建立一個「文文 AI 顧問 preset」或 OpenClaw skill：

- 台灣繁中語境
- 企業顧問欄位
- LINE / Telegram / Google Drive / GitHub Pages 常用架構
- 資安與隱私條款
- 醫療、法務、投資等免責提醒
- Jason 的交付與 B21 備份規則

## 建議角色分工

- 文文：需求訪談、顧問分析、規格整理、驗收標準。
- Spec Kit：產出規格、計畫、任務骨架。
- Coding Agent：依任務實作。
- Sub-agent：負責研究、測試、文件、UI 改善。
- Jason：確認客戶方向、商業承諾、對外溝通。

## 風險與注意事項

- 不要直接把客戶機密資料丟進不受控工具。
- Spec Kit 產出的內容仍需人工審查，不可完全自動承諾。
- 醫療、法務、投資領域只能做資訊整理與流程輔助，不可替代專業判斷。
- 安裝第三方 extensions / presets 前要審查來源。
- 若要在正式客戶案使用，建議先在本機或私有 repo 進行。

## 結論

Spec Kit 很適合變成文文的「AI 顧問技術步驟」延伸模組：

> 先規格，後實作；先驗收，後交付；先治理，後自動化。

這能讓文文從做單次簡報/工具，升級為可以管理企業 AI 導入專案的規格驅動顧問助理。
