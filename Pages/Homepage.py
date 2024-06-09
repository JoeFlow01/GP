import pyttsx3
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
        print("Trying to go to SignLang to speech page, no kv file available")


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


class SpeechToSignLanguagePage(Screen):
    video_playing = False
    screen_manager = screen_manager

    def control_video(self, video_element, play_pause_btn):
        if self.video_playing:
            video_element.state = 'pause'
            self.video_playing = False
            play_pause_btn.text = "Play"
        else:
            video_element.state = 'play'
            self.video_playing = True
            play_pause_btn.text = "Pause"

    def go_to_home(self, video_element, play_pause_btn):
        video_element.state = 'pause'
        play_pause_btn.text = "Play"
        self.screen_manager.current = 'HomePage'

    def change_vid(self, video_element, source):
        video_element.source = source


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
        if voice == 'Female':
            self.engine.setProperty('voice', voices[0].id)
        elif voice == 'Male':
            self.engine.setProperty('voice', voices[1].id)
        else:
            self.engine.setProperty('voice', voices[0].id)

    def clear_text(self):
        self.ids.text_input.text = ''


if __name__ == '__main__':
    Builder.load_file('HomePage.kv')
    Builder.load_file('SpeechToSignLanguagePage.kv')
    Builder.load_file('TextToSpeechPage.kv')
    MyApp().run()
