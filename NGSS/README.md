# NGSS · StarWhisper Telescope

近邻星系巡天（NGSS）上的观测自动化代码，对应论文 [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4)。这是真实观测栈，不是 [`../explore/`](../explore/README.md) 里的合成环境。

运行依赖 NINA、FTP、x-opstep 和望远镜连接。没有这些接口时不要把本地试跑写成硬件闭环已经通了。仓库总览见 [根目录 README](../README.md)。

## Project Structure
```
src/
├── app/          # Core application logic
│   └── app2.py   # Main service entry point
├── module/       # Functional modules
├── script/       # Utility scripts
└── util/         # General utilities
observation Schedule/  # supporting materials, no need to download
runningLog/            # supporting materials, no need to download
```


---

## Service Execution
**Start the NGSS service** from this directory (`NGSS/`):

```bash
uvicorn src.app.app2:app --reload
```

The old `uvicorn main:app2.py` line does not match the file layout. The FastAPI app object lives in `src/app/app2.py`.
Capabilities:
- Observation planning
- Target management (addition/selection)
- NINA automation control
- FTP-based internal network data transfer

---

## Environment Setup
### Python Environment
```bash
conda env create -f observe.yml
```

### Configuration
`observe_config.json` parameters:
```json
{
  "inherit": true,          // Inherit yesterday's schedule
  "time_windows": {         // Target selection time windows (hours)
    "early_night": 0.5,
    "midnight": 2.0,
    "midmorning": 2.0,
    "early_morning": 2.0
  },
  "constraints": {
    "d_moon": 15           // Minimum moon distance (degrees)
  },
  "filters": ["L"],        // Filter configuration
  "exposure": {
    "count": 3,            // Exposure count per target
    "time": 120,           // Exposure duration (seconds)
    "wait": 1.0            // Wait time between exposures (minutes)
  }
}
```

---

## TNS Integration
**Transient Search:**
```bash
python Pachong.py
```
This is a legacy helper for public TNS pages. It can fail when the site or login changes. It is not part of Explore, and it does not command the telescope.

---

## Prerequisites
### 1. Server-Telescope Connection
Configure in:  
`src/module/UdpConnect.py`

### 2. x-opstep Deployment
[X-OPSTEP](https://github.com/Shinehope-Chen/X-OPSTEP/blob/master/README.md) required dependencies:
- astrometry.net
- ASTAP
- Scamp
- SWarp
- HOTPANTS

### 3. NINA Configuration
1. Install [NINA](https://nighttime-imaging.com/)
2. Replace plugin:
```bash
cp FMoraes.NINA.SitesPlugin.dll /path/to/NINA/plugins/
```

### 4. Telescope Connection
Ensure proper connection via NINA's plugin interface

---

## Workflow Automation (n8n)
Example workflows:
- `Make_Observation_Plan.json`
- `NGSS_Agent.json`

Structure principles:
1. LLM-driven decision making
2. FastAPI tool integration
3. Log monitoring at:  
   [http://127.0.0.1:80/check_log](http://127.0.0.1:80/check_log)

---

## LLM Implementation Guide
### Selection Criteria
1. Latest generation model (>7B parameters)
2. Instruct-tuned variants preferred
3. On-premise deployment recommended for:
   - Data security
   - Consistent performance

### Configuration Tips
- Avoid quantized models (<16-bit)
- API alternatives: OpenAI, Anthropic, etc.
- Chinese language support required

---

## Observation Workflow
### Skill 1: Plan Creation
```python
Make_Observation_Plan() → {uuid, log_url}
```

### Skill 2: Plan Review
```python
Get_OB_List(station, date) → observation_plan_url
```

### Skill 3: Plan Execution
```python
Load_Observation_Plan(plan_id) → NINA_integration
```

### Skill 5: Target Addition
```python
Add_Observation_Object(target_params) → confirmation
```

### Skill 6: Transient Handling
```python
Transient_load(station, date, telescope) → analysis_results
```

---

## Operational Constraints
1. Strict parameter validation required
2. Mandatory tool use for all operations
3. Chinese language support required
4. URL outputs must be plaintext (no markdown formatting)

> **Note:** All operations require active connection to telescope control systems
