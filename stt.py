from RealtimeSTT import AudioToTextRecorder

class STT:
    def __init__(self):
        self.recorder = AudioToTextRecorder(language="en", enable_realtime_transcription=False)
        self.move_dir = None

    def get_direction(self):
        return self.move_dir

    def process_text(self, text):
        text = self.clean_text(text)
        print("text detected: ", text)
        if text == "close":
            self.move_dir = "CLOSE"
        elif text == "open":
            self.move_dir = "OPEN"
        elif text == "rotate right":
            self.move_dir = "ROTATE_R"
        elif text == "rotate left":
            self.move_dir = "ROTATE_L"
        elif text == "stop rotation":
            self.move_dir = "ROTATE_STOP"
        else:
            self.move_dir = None
        
    def clean_text(self, text):
        text = text.lower().strip()
        text = text.replace(".", "")
        return text

    def run_stt(self):
        while True:
            self.recorder.text(self.process_text)
