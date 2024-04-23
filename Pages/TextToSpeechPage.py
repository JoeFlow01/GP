# TextToSpeechPage.py

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
import pyttsx3

class BlindDeafCommunicationLayout(BoxLayout):
    def speak(self):
        text = self.ids.text_input.text
        self.engine.say(text)
        self.engine.runAndWait()

class BlindDeafCommunicationApp(App):
    def build(self):
        self.engine = pyttsx3.init()
        return BlindDeafCommunicationLayout()

if __name__ == '__main__':
    Builder.load_file("TextToSpeechPage.kv")
    BlindDeafCommunicationApp().run()
