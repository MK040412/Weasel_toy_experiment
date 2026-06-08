from android_world.env import env_launcher
import argparse
p=argparse.ArgumentParser()
p.add_argument("--adb_path", default="/data2/android-sdk/platform-tools/adb")
p.add_argument("--console_port", type=int, default=5554)
p.add_argument("--grpc_port", type=int, default=8554)
p.add_argument("--perform_emulator_setup", action="store_true")
a=p.parse_args()
env = env_launcher.load_and_setup_env(console_port=a.console_port, emulator_setup=a.perform_emulator_setup, adb_path=a.adb_path, grpc_port=a.grpc_port)
print("ENV_READY", a.console_port, a.grpc_port)
env.close()
