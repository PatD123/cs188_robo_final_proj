from RealtimeSTT import AudioToTextRecorder
from ollama import chat, ChatResponse

class STT:
    def __init__(self):
        self.recorder = AudioToTextRecorder(language="en", enable_realtime_transcription=False)
        self.move_dir = None
        self.prompt = """We are controlling a robot arm using voice commands. The user can rotate the arm, control if the gripper is closed or open, move the arm forward or backward, and change the speed of the arm. Based on the user's input, return the action to be taken. Return only the action, no other text. The action must be one of the following:
        - CLOSE
        - OPEN
        - ROTATE_R
        - ROTATE_L
        - FORWARD
        - BACKWARD
        - SPEED_UP
        - SLOW_DOWN
        - RESET
        - STOP

        Here is the user's input: {text}
        """

    def get_direction(self):
        return self.move_dir

    def process_text(self, text):
        text = self.clean_text(text)
        response = chat(model="llama3.2", messages=[{"role": "user", "content": self.prompt.format(text=text)}])
        self.move_dir = response.message.content
        print("Input: ", text)
        print("Output: ", self.move_dir)
        return self.move_dir

    # def process_text(self, text):
    #     text = self.clean_text(text)
    #     print("text detected: ", text)
    #     if text == "close":
    #         self.move_dir = "CLOSE"
    #     elif text == "open":
    #         self.move_dir = "OPEN"
    #     elif text == "rotate right":
    #         self.move_dir = "ROTATE_R"
    #     elif text == "rotate left":
    #         self.move_dir = "ROTATE_L"
    #     elif text == "forward":
    #         self.move_dir = "FORWARD"
    #     elif text == "back":
    #         self.move_dir = "BACKWARD"
    #     elif text == "increase speed":
    #         self.move_dir = "SPEED_UP"
    #     elif text == "decrease speed":
    #         self.move_dir = "SLOW_DOWN"
    #     elif text == "reset speed":
    #         self.move_dir = "RESET"
    #     elif text == "stop":
    #         self.move_dir = "STOP"
    #     else:
    #         self.move_dir = None
        
    def clean_text(self, text):
        text = text.lower().strip()
        text = text.replace(".", "")
        return text

    def run_stt(self):
        while True:
            self.recorder.text(self.process_text)
