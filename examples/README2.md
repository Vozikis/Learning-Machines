# Instructions for Installing and Running the Simulator

This guide provides step-by-step instructions to set up and run the robot simulator on both macOS and Windows platforms.

---

## Prerequisites

- **Python 3.8**
- **Docker Desktop**
- **CoppeliaSim Educational Version**

---

## 1. Clone the Repository

```shell
# Command for both platforms
git clone https://github.com/ci-group/learning_machines_robobo.git
cd learning_machines_robobo
```

---

## 2. Install Python 3.8

### macOS:

1. Ensure Python 3.8 is installed. Use tools like `pyenv` or `brew` to install the correct version.
2. Create a virtual environment and install dependencies:

   ```shell
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

### Windows:

1. Ensure Python 3.8 is installed and accessible via the `py` launcher.
2. Create a virtual environment and install dependencies:

   ```powershell
   py -3.8 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

---

## 3. Install Docker Desktop

### macOS:

- Install Docker Desktop. Refer to the [Docker Docs for macOS](https://docs.docker.com/desktop/install/mac-install/).
- For Apple Silicon Macs, enable experimental features in Docker settings and limit Docker to 1 CPU core under "Resources" if stability issues occur.

### Windows:

- Install Docker Desktop with WSL2 and enable hardware virtualization.
- Refer to the [Docker Docs for Windows](https://docs.docker.com/desktop/install/windows-install/).

---

## 4. Setup CoppeliaSim

1. Download the educational version of CoppeliaSim from their [official website](https://www.coppeliarobotics.com/downloads).

### macOS:

- Download the appropriate version for your hardware (Intel or Apple Silicon).
- Move the `.app` file to `learning_machines_robobo/examples/full_project_setup/coppeliaSim.app`.
- If macOS blocks the app, use Finder to grant permissions: control-click the app and override the settings.

### Windows:

- Download the zip file for Windows.
- Extract it to `learning_machines_robobo\examples\full_project_setup\CoppeliaSim`.

---

## 5. Configure IP Address

1. Update `scripts/setup.bash` with your machine's IP address.

### macOS:

Run the following command to find your IP address:

```shell
ipconfig getifaddr en0
```

### Windows:

Run the following PowerShell command to find your IP address:

```powershell
(Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
```

Update the `export COPPELIA_SIM_IP` line in `scripts/setup.bash` with your IP address:

```bash
export COPPELIA_SIM_IP="your.ip.address"
```

---

## 6. Start CoppeliaSim

### macOS:

Run the following command in the terminal (ensure your virtual environment is active):

```shell
zsh ./scripts/start_coppelia_sim.zsh ./scenes/Robobo_Scene.ttt
```

### Windows:

Run the following command in PowerShell (ensure your virtual environment is active):

```powershell
./scripts/start_coppelia_sim.ps1 ./scenes/Robobo_Scene.ttt
```

---

## 7. Run the Simulation

### macOS:

- Launch Docker Engine.
- Depending on your hardware, run:

  ```shell
  # Intel Macs
  bash ./scripts/run.sh --simulation

  # Apple Silicon Macs
  zsh ./scripts/run_apple_silicon.zsh --simulation
  ```

### Windows:

- Launch Docker Engine.
- Run the following command:

  ```powershell
  ./scripts/run.ps1 --simulation
  ```

---

## 8. Verify the Setup

If the setup is successful, you should see the Robobo move and output similar to the provided example. Refer to `full_project_setup/catkin_ws/src/learning_machines/scripts/learning_robobo_controller.py` and `test_actions.py` for the executed code.

---

Congratulations! You have successfully set up the robot simulator on your machine.
