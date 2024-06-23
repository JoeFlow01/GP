import pyttsx3
import os
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.animation import Animation
import threading
import speech_recognition as sr
import subprocess

screen_manager = ScreenManager()


class HoverBehavior(object):
    hovered = BooleanProperty(False)
    border_point = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(HoverBehavior, self).__init__(**kwargs)
        self.register_event_type('on_enter')
        self.register_event_type('on_leave')
        Window.bind(mouse_pos=self.on_mouse_pos)

    def on_mouse_pos(self, *args):
        if not self.get_root_window():
            return
        pos = args[1]
        inside = self.collide_point(*self.to_window(*pos))
        if self.hovered == inside:
            return
        self.border_point = pos
        self.hovered = inside
        if inside:
            self.dispatch('on_enter')
        else:
            self.dispatch('on_leave')

    def on_enter(self):
        pass

    def on_leave(self):
        pass


Factory.register('HoverBehavior', cls=HoverBehavior)


class HoverButton(BoxLayout, HoverBehavior):
    def __init__(self, **kwargs):
        super(HoverButton, self).__init__(**kwargs)
        with self.canvas.before:
            self.bg_color = Color(0.6, 0.6, 0.6, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def on_enter(self):
        self.bg_color.rgba = (0.8, 0.8, 0.8, 1)

    def on_leave(self):
        self.bg_color.rgba = (0.65, 0.65, 0.65, 1)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self.action == 'text_to_speech':
                self.go_to_TextToSpeechPage()
            elif self.action == 'speech_to_sign_language':
                self.go_to_SpeechToSignLanguagePage()
            elif self.action == 'sign_language_to_speech':
                self.go_to_SignLanguageToSpeechPage()
            return True
        return super(HoverButton, self).on_touch_down(touch)

    def go_to_TextToSpeechPage(self):
        screen_manager.current = 'TextToSpeechPage'

    def go_to_SpeechToSignLanguagePage(self):
        screen_manager.current = 'SpeechToSignLanguagePage'

    def go_to_SignLanguageToSpeechPage(self):
        print("Sign Language to Speech")
        # screen_manager.current = 'SignLanguageToSpeechPage'
        self.run_main_script()  # Execute main.py script

    def run_main_script(self):
        print("running manin")

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get directory of main script
        subprocess.Popen(["python", os.path.join(script_dir, "main.py")])


class MyApp(App):
    def build(self):
        self.title = "Helping you"
        self.screen_manager = screen_manager
        self.screen_manager.add_widget(SplashScreen(name='SplashScreen'))
        self.screen_manager.add_widget(HomePage(name='HomePage'))
        self.screen_manager.add_widget(SpeechToSignLanguagePage(name='SpeechToSignLanguagePage'))
        self.screen_manager.add_widget(TextToSpeechPage(name='TextToSpeechPage'))
        Clock.schedule_once(self.go_to_home, 2)
        return self.screen_manager

    def go_to_home(self, dt):
        self.screen_manager.current = 'HomePage'


class SplashScreen(Screen):
    pass


class HomePage(Screen):
    pass


class SignLanguageToSpeechPage(Screen):
    screen_manager = screen_manager

    def on_enter(self):
        print("Running main.py")
        subprocess.Popen(["python", "main.py"])

    def go_to_home(self):
        self.screen_manager.current = 'HomePage'


class SpeechToSignLanguagePage(Screen):
    video_playing = False
    screen_manager = screen_manager

    def recognize_speech_from_mic(self, video_element, displayer_label):
        def update_label_to_listening(dt):
            displayer_label.text = "Listening"
            # Start the speech recognition in a separate thread to avoid blocking the UI

        # Pause the video element
        video_element.state = 'pause'
        # Schedule the label update function to run after 0.1 seconds
        Clock.schedule_once(update_label_to_listening, 0.1)
        threading.Thread(target=self.start_listening, args=(video_element, displayer_label)).start()

    def start_listening(self, video_element, displayer_label):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            try:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                recognizer.energy_threshold = 4000
                audio = recognizer.listen(mic, timeout=5, phrase_time_limit=5)
                print("Audio recording done")
                try:
                    text = recognizer.recognize_google(audio)
                    displayer_label.text = f"Recognized text: {text}"
                    text = text.lower()
                    text = text.replace(" ", "")

                    if text in ["eat", "hello", "help", "iloveyou", "no", "sorry", "thankyou", "yes"]:
                        source = "../Videos/" + text + ".mp4"
                    else:
                        source = "../Assets/invalid.jpeg"

                    video_element.source = source
                    video_element.state = 'play'

                    print(f"Recognized text: {text}")
                    return text

                except sr.UnknownValueError:
                    displayer_label.text = "Google Speech Recognition could not understand audio"
                    print("Google Speech Recognition could not understand audio")
                    return "Could not understand audio"
                except sr.RequestError as e:
                    displayer_label.text = f"Could not request results; {e}"
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                    return "API unavailable"
            except sr.WaitTimeoutError:
                displayer_label.text = 'Listening timed out while waiting for phrase to start'
                print('Listening timed out while waiting for phrase to start')
                return "Listening timed out"
            except Exception as e:
                displayer_label.text = f"An error occurred: {e}"
                print(f"An error occurred: {e}")
                return "An error occurred"

    def go_to_home(self, video_element):
        video_element.state = 'pause'
        self.screen_manager.current = 'HomePage'


class TextToSpeechPage(Screen):
    engine = pyttsx3.init()
    screen_manager = screen_manager

    def __init__(self, **kwargs):
        super(TextToSpeechPage, self).__init__(**kwargs)
        self.engine.connect('started-utterance', self.on_speech_start)
        self.engine.connect('finished-utterance', self.on_speech_end)
        self.animation = None

    def go_to_home(self):
        self.screen_manager.current = 'HomePage'

    def speak(self):
        text_input = self.ids.text_input
        text = text_input.text
        if text:
            self.engine.say(text)
            threading.Thread(target=self.engine.runAndWait).start()
        else:
            text = "Please enter text"
            self.engine.say(text)
            threading.Thread(target=self.engine.runAndWait).start()

    def on_speech_start(self, name):
        self.show_speech_feedback()

    def on_speech_end(self, name, completed):
        self.stop_speech_feedback()

    def show_speech_feedback(self):
        label = self.ids.speech_feedback
        icon = self.ids.sound_icon
        self.animation = Animation(opacity=1, duration=0.5) + Animation(opacity=0, duration=0.5)
        self.animation.repeat = True
        self.animation.start(label)
        icon.opacity = 1

    def stop_speech_feedback(self):
        if self.animation:
            self.animation.stop(self.ids.speech_feedback)
            self.ids.speech_feedback.opacity = 0
        self.ids.sound_icon.opacity = 0

    def set_voice(self, voice):
        voices = self.engine.getProperty('voices')
        if voice == 'Voice 1':
            print("v1")
            self.engine.setProperty('voice', voices[0].id)
        elif voice == 'Voice 2':
            print("v2")
            self.engine.setProperty('voice', voices[1].id)
        else:
            print("def")
            self.engine.setProperty('voice', voices[0].id)

    def clear_text(self):
        self.ids.text_input.text = ''


if __name__ == '__main__':
    Builder.load_file('HomePage.kv')
    Builder.load_file('SpeechToSignLanguagePage.kv')
    Builder.load_file('TextToSpeechPage.kv')
    MyApp().run()
