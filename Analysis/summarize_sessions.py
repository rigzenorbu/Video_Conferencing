import os
import csv
import glob

def get_session_info(filepath):
    try:
        with open(filepath, 'rb') as f:
            # Get first line (after header)
            f.readline() # skip header
            first_line = f.readline().decode('utf-8').strip()
            if not first_line:
                return 0, 0
            start_ns = int(first_line.split(',')[0])
            
            # Fast seek to the end to get the last packet
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(max(file_size - 1000, 0))
            lines = f.readlines()
            
            last_line = lines[-1].decode('utf-8').strip()
            if not last_line and len(lines) > 1:
                last_line = lines[-2].decode('utf-8').strip()
                
            end_ns = int(last_line.split(',')[0])
            duration_sec = (end_ns - start_ns) / 1_000_000_000
            size_mb = file_size / (1024 * 1024)
            
            return duration_sec, size_mb
    except Exception as e:
        return 0, 0

def summarize_platform(directory):
    if not os.path.exists(directory):
        print(f"[WARN] Directory {directory} not found.")
        return
        
    files = glob.glob(os.path.join(directory, "*.csv"))
    print(f"\n================================================================================")
    print(f" PLATFORM: {os.path.basename(directory).upper()} | TOTAL SESSIONS: {len(files)}")
    print(f"================================================================================")
    
    sessions = []
    for f in files:
        dur, size = get_session_info(f)
        # Only keep sessions that lasted longer than 10 seconds (filters out instant disconnects)
        if dur > 10:
            sessions.append((os.path.basename(f), dur, size))
    
    # Sort by duration descending
    sessions.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Session Filename':<48} | {'Duration':<12} | {'File Size'}")
    print("-" * 80)
    for name, dur, size in sessions:
        mins = int(dur // 60)
        secs = int(dur % 60)
        print(f"{name:<48} | {mins:02d}m {secs:02d}s".ljust(65) + f"| {size:.2f} MB")
    print("\n")

if __name__ == "__main__":
    # Pointing to the folder you just moved here
    base_dir = "Sessions_18_to_19"
    
    summarize_platform(os.path.join(base_dir, "Teams"))
    summarize_platform(os.path.join(base_dir, "Meet"))
