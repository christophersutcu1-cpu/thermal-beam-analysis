import telnetlib
import time

# Controller Configuration
CTRL_IP = "192.168.10.5"
PORT = 23
TIMEOUT = 5

def run_step_experiment():
    try:
        # 1. Establish Connection
        print(f"Connecting to controller at {CTRL_IP}...")
        tn = telnetlib.Telnet(CTRL_IP, PORT, timeout=TIMEOUT)
        
        # Clear the initial greeting buffer
        time.sleep(0.5)
        tn.read_very_eager()

        # 2. RESET: Force trigger line to 0
        print("Step 1: Resetting trigger line to 0...")
        tn.write(b"c0;\r\n")
        time.sleep(1) 
        tn.read_very_eager()

        # 3. TRIGGER: Start ResearchIR Recording
        print("Step 2: Triggering Camera Recording (c1:0;)...")
        tn.write(b"c1:0;\r\n") 
        
        # 4. PRE-BUFFER (Cold Baseline)
        print("Capturing 1s cold baseline...")
        time.sleep(1)

        # 5. LAMPS ON (1 Second Pulse @ 500W)
        # Note: d128 represents ~50% duty cycle for 500W output on 1000W lamps
        print("Step 3: Switching LAMPS ON for 1s at 500W (d127;)...")
        tn.write(b"d127;\r\n")
        time.sleep(10) # <--- UPDATED: 1 second pulse

        # 6. LAMPS OFF
        print("Step 4: Switching LAMPS OFF (d0;)...")
        tn.write(b"d0;\r\n")

        # 7. COOL-DOWN OBSERVATION (20 Second Recording)
        print("Step 5: Commencing 20s cool-down observation...")
        time.sleep(20) # <--- UPDATED: Corrected to match your 20s requirement

        # 8. STOP RECORDING
        print("Step 6: Resetting trigger line to 0. Sequence complete.")
        tn.write(b"c0;\r\n")
        time.sleep(0.5)

        # 9. CLOSE SESSION
        tn.write(b"q\r\n")
        tn.close()
        print("Done. Connection closed.")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    run_step_experiment()