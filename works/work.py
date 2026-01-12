import pandas as pd
import subprocess
import platform
import tkinter as tk
from tkinter import ttk
import threading
import queue
import time


class PingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ping Status Dashboard")
        self.root.geometry("800x600")

        self.results_queue = queue.Queue()
        self.items = {}  # To store treeview item IDs by vpn_ip

        self.create_widgets()
        self.load_data()

        self.start_ping_thread()
        self.process_queue()

    def create_widgets(self):
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)
        style.configure("Treeview.Heading", font=("Calibri", 10, "bold"))

        self.tree = ttk.Treeview(
            self.root,
            columns=("Name", "IP(option)", "VPN IP", "Status"),
            show="headings",
        )
        self.tree.heading("Name", text="Name")
        self.tree.heading("IP(option)", text="IP(option)")
        self.tree.heading("VPN IP", text="VPN IP")
        self.tree.heading("Status", text="Status")

        self.tree.column("Name", width=250)
        self.tree.column("IP(option)", width=150)
        self.tree.column("VPN IP", width=150)
        self.tree.column("Status", width=100, anchor=tk.CENTER)

        # Tags for status colors
        self.tree.tag_configure("success", background="lightgreen")
        self.tree.tag_configure("failed", background="lightcoral")
        self.tree.tag_configure("pending", background="lightgrey")

        vsb = ttk.Scrollbar(self.root, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

    def load_data(self):
        csv_file_path = "251117-1212 AP of WONJU City Hall (1).csv"
        try:
            # df = pd.read_csv(csv_file_path, encoding="ansi")
            df = pd.read_csv(csv_file_path)
            required_columns = ["vpn ip", "ip(option)", "name"]
            if not all(col in df.columns for col in required_columns):
                self.update_status_display("Error: Missing required columns in CSV.")
                return

            for index, row in df.iterrows():
                vpn_ip = str(row.get("vpn ip", "")).strip()
                if vpn_ip:
                    name = str(row.get("name", "N/A")).strip()
                    ip_option = str(row.get("ip(option)", "N/A")).strip()

                    item_id = self.tree.insert(
                        "",
                        "end",
                        values=(name, ip_option, vpn_ip, "Pending"),
                        tags=("pending",),
                    )
                    self.items[vpn_ip] = item_id

        except FileNotFoundError:
            self.update_status_display(f"Error: CSV file '{csv_file_path}' not found.")
        except Exception as e:
            self.update_status_display(f"Error reading CSV: {e}")

    def update_status_display(self, message):
        # Fallback to show error if treeview fails
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", "end", values=(message, "", "", ""))

    def start_ping_thread(self):
        self.ping_thread = threading.Thread(target=self.ping_worker, daemon=True)
        self.ping_thread.start()

    def ping_worker(self):
        # Determine ping command and creation flags based on platform
        if platform.system().lower() == "windows":
            base_command = ["ping", "-n", "1"]
            creation_flags = subprocess.CREATE_NO_WINDOW
        else:
            base_command = ["ping", "-c", "1"]
            creation_flags = 0  # No specific flags for other platforms

        while True:
            for vpn_ip, item_id in self.items.items():
                status = "failed"
                try:
                    # Add the IP address to the command
                    command = base_command + [vpn_ip]
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=1,
                        creationflags=creation_flags,  # Add this flag
                    )
                    if result.returncode == 0:
                        status = "success"
                except (subprocess.TimeoutExpired, Exception):
                    status = "failed"

                self.results_queue.put((item_id, status))

            time.sleep(2)  # Wait 2 seconds before the next round of pings

    def process_queue(self):
        try:
            while not self.results_queue.empty():
                item_id, status = self.results_queue.get_nowait()
                current_values = self.tree.item(item_id, "values")
                self.tree.item(
                    item_id,
                    values=(
                        current_values[0],
                        current_values[1],
                        current_values[2],
                        status.capitalize(),
                    ),
                    tags=(status,),
                )
        finally:
            self.root.after(100, self.process_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = PingApp(root)
    root.mainloop()
