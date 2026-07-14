import os
import glob
from datetime import datetime

def get_session_info(filepath):
    try:
        with open(filepath, 'rb') as f:
            f.readline() # skip header
            first_line = f.readline().decode('utf-8').strip()
            if not first_line: return None
            start_ns = int(first_line.split(',')[0])
            
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(max(file_size - 1000, 0))
            lines = f.readlines()
            last_line = lines[-1].decode('utf-8').strip()
            if not last_line and len(lines) > 1:
                last_line = lines[-2].decode('utf-8').strip()
            end_ns = int(last_line.split(',')[0])
            
            # Extract IP from filename (e.g., Session_10.184.33.232_1781...csv)
            basename = os.path.basename(filepath)
            parts = basename.split('_')
            # Extract everything between 'Session_' and the timestamp
            ip_part = basename[8:-len(parts[-1])-1] 
            
            dur = (end_ns - start_ns) / 1_000_000_000
            
            # Convert nanoseconds to human readable time
            start_dt = datetime.fromtimestamp(start_ns / 1_000_000_000).strftime('%b %d, %H:%M:%S')
            end_dt = datetime.fromtimestamp(end_ns / 1_000_000_000).strftime('%b %d, %H:%M:%S')
            
            return {
                'ip': ip_part,
                'start_dt': start_dt,
                'end_dt': end_dt,
                'start_ns': start_ns,
                'duration': dur
            }
    except Exception as e:
        return None

files = glob.glob("Sessions_18_to_19/Meet/*.csv")
sessions_by_ip = {}

for f in files:
    info = get_session_info(f)
    if info and info['duration'] > 10:  # Ignore phantom 1-second glitches
        ip = info['ip']
        if ip not in sessions_by_ip:
            sessions_by_ip[ip] = []
        sessions_by_ip[ip].append(info)

# Filter for IPs that have more than 1 session
repeated_ips = {ip: sessions for ip, sessions in sessions_by_ip.items() if len(sessions) > 1}

print("\n=========================================================================================")
print(f" TEAMS REPEATED SESSIONS TIMELINE (Found {len(repeated_ips)} distinct IPs with multiple sessions)")
print("=========================================================================================")
print(f"{'Client IP':<25} | {'Session':<7} | {'Start Time':<18} | {'End Time':<18} | {'Duration'}")
print("-" * 89)

# Sort IPs by how many times they repeated (most to least)
sorted_ips = sorted(repeated_ips.items(), key=lambda x: len(x[1]), reverse=True)

for ip, sessions in sorted_ips:
    # Sort the sessions for this specific IP chronologically
    sessions.sort(key=lambda x: x['start_ns'])
    
    for idx, s in enumerate(sessions):
        mins = int(s['duration'] // 60)
        secs = int(s['duration'] % 60)
        dur_str = f"{mins:02d}m {secs:02d}s"
        
        # Only print the IP address on the first line for clean formatting
        ip_display = ip if idx == 0 else ""
        print(f"{ip_display:<25} | #{idx+1:<6} | {s['start_dt']:<18} | {s['end_dt']:<18} | {dur_str}")
    
    # Add a divider between different users
    print("-" * 89)
print("\n")
