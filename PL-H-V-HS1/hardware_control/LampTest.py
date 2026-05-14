import telnetlib
import time

# --- CONFIGURATION ---
HOST = "192.168.10.5"
PORT = 23

def flash_lamps():
    try:
        # 1. Connect to the IRT Controller
        tn = telnetlib.Telnet(HOST, PORT, timeout=5)
        print(f"Connected to Lamp Controller at {HOST}")

        # 2. Pre-lamp delay (1 second)
        print("Waiting 1 second...")
        time.sleep(1)

        # 3. Lamps ON (Power level 255 = 100%)
        tn.write(b"d255;\n")
        print("Lamps ON")

        # 4. Hold for 2 seconds
        time.sleep(2)

        # 5. Lamps OFF
        tn.write(b"d0;\n")
        print("Lamps OFF")

        # 6. Close session
        tn.write(b"q\n")
        tn.close()
        print("Done.")

    except Exception as e:
        print(f"Failed to connect: {e}")
        print("Check that your laptop is set to 192.168.10.10 and plugged into ETH3.")

if __name__ == "__main__":
    flash_lamps()