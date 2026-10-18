import ctypes, threading, tkinter as tk, sys

# 1. Global Hook Constants and Setup (Win32 API)
WH_KEYBOARD_LL = 13  # Low-level keyboard hook ID
WM_KEYDOWN     = 0x0100
WM_KEYUP       = 0x0101
WM_SYSKEYDOWN  = 0x0104
WM_SYSKEYUP    = 0x0105

# Virtual-Key codes for modifiers
VK_SHIFT   = 0x10
VK_CONTROL = 0x11
VK_MENU    = 0x12   # Alt key
VK_CAPITAL = 0x14   # Caps Lock
VK_LWIN    = 0x5B
VK_RWIN    = 0x5C

# Define ctypes types for WinAPI structures
WORD    = ctypes.c_ushort
DWORD   = ctypes.c_uint32
ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

class KBDLLHOOKSTRUCT(ctypes.Structure):
    fields = [("vkCode", DWORD),
                ("scanCode", DWORD),
                ("flags", DWORD),
                ("time", DWORD),
                ("dwExtraInfo", ULONG_PTR)]

class KEYBDINPUT(ctypes.Structure):
    fields = [("wVk",  WORD),
                ("wScan", WORD),
                ("dwFlags", DWORD),
                ("time",   DWORD),
                ("dwExtraInfo", ULONG_PTR)]

class MOUSEINPUT(ctypes.Structure):
    fields = [("dx",       DWORD),
                ("dy",       DWORD),
                ("mouseData",DWORD),
                ("dwFlags",  DWORD),
                ("time",     DWORD),
                ("dwExtraInfo", ULONG_PTR)]

class HARDWAREINPUT(ctypes.Structure):
    fields = [("uMsg",    DWORD),
                ("wParamL", WORD),
                ("wParamH", WORD)]

class INPUT_UNION(ctypes.Union):
    fields = [("ki", KEYBDINPUT),
                ("mi", MOUSEINPUT),
                ("hi", HARDWAREINPUT)]

class INPUT(ctypes.Structure):
    fields = [("type", DWORD),
                ("union", INPUT_UNION)]
    anonymous = ("union",)

class MSG(ctypes.Structure):
    fields = [("hwnd",   ctypes.c_void_p),
                ("message", ctypes.c_uint),
                ("wParam", ctypes.c_void_p),
                ("lParam", ctypes.c_void_p),
                ("time",   ctypes.c_uint),
                ("pt_x",   ctypes.c_long),
                ("pt_y",   ctypes.c_long)]

# Load WinAPI functions
user32   = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Set argument and return types for needed WinAPI functions
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
user32.SetWindowsHookExW.restype  = ctypes.c_void_p
user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
user32.UnhookWindowsHookEx.restype  = ctypes.c_bool
user32.CallNextHookEx.argtypes  = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
user32.CallNextHookEx.restype   = ctypes.c_int
user32.GetMessageW.argtypes     = [ctypes.POINTER(MSG), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
user32.GetMessageW.restype      = ctypes.c_int
user32.TranslateMessage.argtypes= [ctypes.POINTER(MSG)]
user32.TranslateMessage.restype = ctypes.c_bool
user32.DispatchMessageW.argtypes= [ctypes.POINTER(MSG)]
user32.DispatchMessageW.restype = ctypes.c_int
user32.PostThreadMessageW.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
user32.PostThreadMessageW.restype  = ctypes.c_bool
user32.GetKeyState.argtypes    = [ctypes.c_int]
user32.GetKeyState.restype     = ctypes.c_short
user32.SendInput.argtypes      = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype       = ctypes.c_uint

# 2. Character substitution mapping for dyslexia adaptation
# Unicode alternatives for letters (increasing "morphic transformation" with intensity)
map_intensity = {
    1: {'b': 'ʙ', 'd': 'ð'},  # minimal substitutions
    2: {'b': 'ʙ', 'd': 'ð', 'p': 'þ', 'q': 'ɋ'},
    3: {'b': 'ʙ', 'd': 'ð', 'p': 'þ', 'q': 'ɋ', 'n': 'ɴ', 'u': 'ᴜ'},
    4: {'b': 'ʙ', 'd': 'ð', 'p': 'þ', 'q': 'ɋ', 'n': 'ɴ', 'u': 'ᴜ', 'm': 'ᴍ', 'w': 'ᴡ'},
    5: {'b': 'ʙ', 'd': 'ð', 'p': 'þ', 'q': 'ɋ', 'n': 'ɴ', 'u': 'ᴜ', 'm': 'ᴍ', 'w': 'ᴡ', 'i': 'ɪ', 'l': 'ʟ'}
}

# 3. Global variables for hook state
hook_id = None            # handle for the installed hook
hook_thread_id = None     # thread ID of the hook (for posting messages)
hook_enabled = False      # flag to activate/deactivate the hook dynamically
current_intensity = 1     # current intensity level (1-5)
suppressed_keys = set()   # set of key codes suppressed (to handle key-up events)
MY_EXTRA_INFO = 0xD17A59E1  # custom identifier for our injected events (any unique value)

# 4. Hook callback function (LowLevelKeyboardProc) in C types
@ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
def low_level_proc(nCode, wParam, lParam):
    """Low-level keyboard hook procedure to intercept and modify keystrokes."""
    global hook_enabled, current_intensity, suppressed_keys
    if nCode < 0:
        # Pass the event to next hook if nCode is negative (no processing needed)
        return user32.CallNextHookEx(hook_id, nCode, wParam, lParam)
    # Get keyboard event data
    kb_data = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
    vk = kb_data.vkCode  # virtual key code of the key event

    # Handle key-up events (to release suppressed keys)
    if wParam in (WM_KEYUP, WM_SYSKEYUP):
        if vk in suppressed_keys:
            # Remove from suppressed set and block this key-up from the target application
            suppressed_keys.discard(vk)
            user32.CallNextHookEx(hook_id, nCode, wParam, lParam)  # allow other hooks to process
            return 1  # block from further processing (do not send to application)
        # Not a suppressed key, let it pass through normally
        return user32.CallNextHookEx(hook_id, nCode, wParam, lParam)

    # Handle key-down events
    if wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
        # Ignore events injected by our own SendInput (identified by flag and custom extra info)
        if (kb_data.flags & 0x10) and kb_data.dwExtraInfo == MY_EXTRA_INFO:
            return user32.CallNextHookEx(hook_id, nCode, wParam, lParam)
        # If hook is disabled, do not transform, just pass through
        if not hook_enabled:
            return user32.CallNextHookEx(hook_id, nCode, wParam, lParam)
        # Skip transformation if a modifier (Ctrl/Alt/Win) is pressed to avoid breaking shortcuts
        ctrl_down = (user32.GetKeyState(VK_CONTROL) & 0x8000) != 0
        alt_down  = (kb_data.flags & 0x20) != 0 or ((user32.GetKeyState(VK_MENU) & 0x8000) != 0)
        win_down  = (user32.GetKeyState(VK_LWIN) & 0x8000) != 0 or ((user32.GetKeyState(VK_RWIN) & 0x8000) != 0)
        if ctrl_down or alt_down or win_down:
            return user32.CallNextHookEx(hook_id, nCode, wParam, lParam)
        # Only intercept letter keys (A-Z)
        if 0x41 <= vk <= 0x5A:
            # Determine the actual character (considering Shift and CapsLock state)
            shift_pressed = (user32.GetKeyState(VK_SHIFT) & 0x8000) != 0
            caps_on = (user32.GetKeyState(VK_CAPITAL) & 0x0001) != 0
            # Compute the character the user intended to type
            char = chr(vk)  # this gives uppercase letter by default
            if caps_on ^ shift_pressed:
                # One of CapsLock or Shift is on (exclusive) -> intended uppercase
                char_out = char
            else:
                # Neither or both are on -> intended lowercase
                char_out = char.lower()
            # Look up substitution for this character at the current intensity
            sub_map = map_intensity.get(current_intensity, {})
            ch_lower = char_out.lower()
            if ch_lower in sub_map:
                # We have a substitution for this letter
                new_char = sub_map[ch_lower]
                # Preserve case: if original intended char was uppercase, convert replacement to uppercase if possible
                if char_out.isupper():
                    try:
                        new_char = new_char.upper()
                    except AttributeError:
                        pass
                # Suppress the original key event
                suppressed_keys.add(vk)
                user32.CallNextHookEx(hook_id, nCode, wParam, lParam)  # let other hooks process it
                # Inject the new Unicode character in place of the original
                # Prepare two INPUT events: keydown and keyup for the Unicode character
                input_list = (INPUT * 2)()
                # Key down event for Unicode char
                input_list[0].type = 1  # INPUT_KEYBOARD
                input_list[0].ki = KEYBDINPUT(0, ord(new_char), 0x0004, 0, MY_EXTRA_INFO)  # KEYEVENTF_UNICODE (0x0004)
                # Key up event for Unicode char
                input_list[1].type = 1  # INPUT_KEYBOARD
                input_list[1].ki = KEYBDINPUT(0, ord(new_char), 0x0004 | 0x0002, 0, MY_EXTRA_INFO)  # KEYEVENTF_UNICODE | KEYEVENTF_KEYUP (0x0002)
                user32.SendInput(2, ctypes.byref(input_list), ctypes.sizeof(INPUT))
                # Block the original key from reaching the application
                return 1
    # For all other events, call next hook in chain without blocking
    return user32.CallNextHookEx(hook_id, nCode, wParam, lParam)

# 5. Hook thread function to install and maintain the global hook
def hook_thread_func():
    """Thread procedure to install the global keyboard hook and run a message loop."""
    global hook_id, hook_thread_id
    # Install the low-level keyboard hook
    h_instance = kernel32.GetModuleHandleW(None)
    hook_id = user32.SetWindowsHookExW(WH_KEYBOARD_LL, low_level_proc, h_instance, 0)
    if not hook_id:
        err = kernel32.GetLastError()
        ctypes.windll.user32.MessageBoxW(None, f"Failed to install hook (Error {err}).", "Hook Error", 0)
        return
    hook_thread_id = kernel32.GetCurrentThreadId()
    # Message loop to keep the hook active
    msg = MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))
    # Unhook upon exiting the message loop
    if hook_id:
        user32.UnhookWindowsHookEx(hook_id)
        hook_id = None

# 6. Launch the hook thread (initially keep hook disabled until user activates)
hook_thread = threading.Thread(target=hook_thread_func, daemon=True)
hook_thread.start()
hook_enabled = False

# 7. GUI Application using Tkinter
root = tk.Tk()
root.title("Dyslexia Text Adapter")

# Frames for Calibration and Main settings
frame_calib = tk.Frame(root, padx=10, pady=10)
frame_main  = tk.Frame(root, padx=10, pady=10)
frame_calib.pack()

# Calibration UI elements
calib_label = tk.Label(frame_calib, text="Calibration: which text is easier to read?")
calib_label.pack(pady=5)
optionA_label = tk.Label(frame_calib, text="", font=("Arial", 12))
optionB_label = tk.Label(frame_calib, text="", font=("Arial", 12))
optionA_label.pack(pady=5)
optionB_label.pack(pady=5)
btnA = tk.Button(frame_calib, text="Prefer Option A")
btnB = tk.Button(frame_calib, text="Prefer Option B")
btnA.pack(side="left", padx=20, pady=10)
btnB.pack(side="right", padx=20, pady=10)

# Sample text for comparison (contains various letters)
sample_text = "The quick brown fox jumps over the lazy dog."

calib_step = 0
first_choice = None

def start_calibration():
    """Begin the two-step preference calibration."""
    global calib_step
    calib_step = 1
    # Step 1: Normal text vs Moderately modified text
    optionA_label.config(text=f"Option A (Normal): {sample_text}")
    # Generate text with intensity 3 modifications for Option B
    mod_map = map_intensity[3]
    modified_text = "".join(mod_map.get(ch.lower(), ch) if ch.isalpha() else ch for ch in sample_text)
    optionB_label.config(text=f"Option B (Modified): {modified_text}")
    # Set button commands for first choice
    btnA.config(command=lambda: choose_calib_option('A'))
    btnB.config(command=lambda: choose_calib_option('B'))

def choose_calib_option(choice):
    """Handle user's choice in calibration steps."""
    global calib_step, first_choice, current_intensity
    if calib_step == 1:
        # Record the first comparison choice
        first_choice = choice  # 'A' or 'B'
        calib_step = 2
        # Prepare step 2 based on first choice
        if first_choice == 'A':
            # User preferred normal text; now compare Normal vs Slightly Modified
            optionA_label.config(text=f"Option A (Normal): {sample_text}")
            slight_map = map_intensity[1]
            slight_text = "".join(slight_map.get(ch.lower(), ch) if ch.isalpha() else ch for ch in sample_text)
            optionB_label.config(text=f"Option B (Slightly Modified): {slight_text}")
        else:
            # User preferred modified text; now compare Moderate vs Heavily Modified
            moderate_map = map_intensity[3]
            heavy_map = map_intensity[5]
            moderate_text = "".join(moderate_map.get(ch.lower(), ch) if ch.isalpha() else ch for ch in sample_text)
            heavy_text    = "".join(heavy_map.get(ch.lower(), ch) if ch.isalpha() else ch for ch in sample_text)
            optionA_label.config(text=f"Option A (Moderate): {moderate_text}")
            optionB_label.config(text=f"Option B (Heavy): {heavy_text}")
        # Update button commands for second choice
        btnA.config(command=lambda: choose_calib_option('A2'))
        btnB.config(command=lambda: choose_calib_option('B2'))
    elif calib_step == 2:
        # Determine final intensity based on two answers
        if first_choice == 'A':
            # User found normal easier than moderate in step 1
            # Final intensity = 1 (minimal transformation), since they did not favor heavier changes
            current_intensity = 1
        else:
            # User found modified easier in step 1
            if choice == 'B2':
                # They preferred heavy changes over moderate -> use highest intensity
                current_intensity = 5
            else:
                # They preferred moderate over heavy -> use medium intensity
                current_intensity = 3
        # Switch to main settings interface
        frame_calib.pack_forget()
        show_main_interface()

def show_main_interface():
    """Display the main interface with intensity control and hook toggle."""
    frame_main.pack()
    # Intensity slider setup
    tk.Label(frame_main, text="Intensity level:").grid(row=0, column=0, sticky="w", pady=5)
    intensity_slider = tk.Scale(frame_main, from_=1, to=5, orient="horizontal", length=200)
    intensity_slider.set(current_intensity)
    intensity_slider.grid(row=0, column=1, sticky="w", padx=5, pady=5)
    def on_intensity_change(val):
        # Update current_intensity from slider
        global current_intensity
        current_intensity = int(val)
    intensity_slider.configure(command=on_intensity_change)
    # Hook toggle button
    toggle_btn = tk.Button(frame_main, text="Activate", width=12)
    def toggle_hook():
        global hook_enabled
        if not hook_enabled:
            hook_enabled = True
            toggle_btn.config(text="Deactivate")
        else:
            hook_enabled = False
            toggle_btn.config(text="Activate")
        # Update status label immediately when toggled
        status_label.config(text=f"Hook is currently {'ON' if hook_enabled else 'OFF'}")
    toggle_btn.config(command=toggle_hook)
    toggle_btn.grid(row=1, column=0, columnspan=2, pady=10)
    # Status label to show hook state
    status_label = tk.Label(frame_main, text="Hook is currently OFF")
    status_label.grid(row=2, column=0, columnspan=2)
    # Periodically refresh the status label (in case intensity or state changes)
    def refresh_status():
        status_label.config(text=f"Hook is currently {'ON' if hook_enabled else 'OFF'}")
        root.after(500, refresh_status)
    refresh_status()

# 8. Administrator privilege check (for installing a global hook)
try:
    is_admin = ctypes.windll.shell32.IsUserAnAdmin()
except:
    is_admin = False
if not is_admin:
    # Attempt to restart the script as administrator
    try:
        script_path = sys.argv[0]
        params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script_path}" {params}', None, 1)
        sys.exit(0)
    except Exception:
        # If elevation failed, warn the user (the hook may not work without admin rights)
        ctypes.windll.user32.MessageBoxW(None, 
            "Warning: Administrator privileges might be required for the hook to function.", 
            "Privilege Warning", 0)

# 9. Start the calibration process and main Tkinter loop
start_calibration()
root.protocol("WM_DELETE_WINDOW", lambda: (user32.PostThreadMessageW(hook_thread_id, 0x0012, None, None), root.destroy()))
root.mainloop()