import time
from string import Template

import httpx
from pynput import keyboard
from pynput.keyboard import Key, Controller
import pyperclip


controller = Controller()

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_CONFIG = {
    "model": "mistral:7b-instruct-q4_K_S",
    "keep_alive": "5m",
    "stream": False,
}

PROMPT_TEMPLATE = Template(
    """Fix all typos and casing and punctuation in this text, but preserve all new line characters:

$text

Return only the corrected text, don't include a preamble.
"""
)


def fix_text(text):
    print("Try fixing typos...")
    prompt = PROMPT_TEMPLATE.substitute(text=text)
    response = httpx.post(
        OLLAMA_ENDPOINT,
        json={"prompt": prompt, **OLLAMA_CONFIG},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    if response.status_code != 200:
        print("Error", response.status_code)
        return None
    return response.json()["response"].strip()


def fix_current_line():
    # macOS short cut to select current line: Cmd+Shift+Left
    controller.press(Key.ctrl)
    controller.press(Key.shift)
    controller.press(Key.up)

    controller.release(Key.ctrl)
    controller.release(Key.shift)
    controller.release(Key.up)

    fix_selection()


def fix_selection():

    # 1. Copy selection to clipboard
    with controller.pressed(Key.ctrl):
        controller.tap("a")
        time.sleep(0.1)
        controller.tap("c")

    # 2. Get the clipboard string
    time.sleep(0.1)
    text = pyperclip.paste()

    # 3. Fix string
    if not text:
        return
    fixed_text = fix_text(text)
    if not fixed_text:
        return

    # 4. Paste the fixed string to the clipboard
    pyperclip.copy(fixed_text)
    time.sleep(0.1)

    # 5. Paste the clipboard and replace the selected text
    with controller.pressed(Key.ctrl):
        controller.tap("v")


def on_f9():
    fix_current_line()

def on_f10():
    fix_selection()


def main():
    with keyboard.GlobalHotKeys({"<120>": on_f9, "<121>": on_f10}) as h:
        h.join()

if __name__ == "__main__":
    main()



    

