from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
import pyttsx3

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
            return  # Widget is not displayed
        pos = args[1]
        inside = self.collide_point(*self.to_window(*pos))
        if self.hovered == inside:
            return  # The hover state has not changed
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
            self.bg_color = Color(1, 1, 1, 1)  # Set background color to white
            self.rect = Rectangle(pos=self.pos, size=self.size)

        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def on_enter(self):
        self.bg_color.rgba = (0.8, 0.8, 0.8, 1)  # Change to light grey on hover

    def on_leave(self):
        self.bg_color.rgba = (1, 1, 1, 1)  # Change back to white when not hovered

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Check the action property and perform the corresponding action
            if self.action == 'text_to_speech':
                self.go_to_TextToSpeechPage()  # Call the appropriate method
            elif self.action == 'speech_to_sign_language':
                self.go_to_SpeechToSignLanguagePage()  # Call the appropriate method
            elif self.action == 'sign_language_to_speech':
                self.go_to_SignLanguageToSpeechPage()  # Call the appropriate method
            return True
        return super(HoverButton, self).on_touch_down(touch)

    def go_to_TextToSpeechPage(self):
        print("Text To Speeech")
        screen_manager.current = 'TextToSpeechPage'

    def go_to_SpeechToSignLanguagePage(self):
        print("Speech to SignLang")
        screen_manager.current = 'SpeechToSignLanguagePage'

    def go_to_SignLanguageToSpeechPage(self):
        print("Trying to go to SingLang to speech page,no kv file available ")


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

    def go_to_home(self,video_element,play_pause_btn):
        video_element.state = 'pause'
        play_pause_btn.text = "Play"
        self.screen_manager.current = 'HomePage'

    def change_vid(self, video_element, source):
        video_element.source = source


class TextToSpeechPage(Screen):
    engine = pyttsx3.init()
    screen_manager = screen_manager

    def go_to_home(self):
        self.screen_manager.current = 'HomePage'

    def speak(self):
        text_input = self.ids.text_input  # Accessing the text input widget
        text = text_input.text  # Getting the text from the text input
        if text:
            print("Text entered:", text)  # Print the text for testing
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            text = "Please enter text "
            print("Text entered:", text)  # Print the text for testing
            self.engine.say(text)
            self.engine.runAndWait()


if __name__ == '__main__':
    Builder.load_file('HomePage.kv')
    Builder.load_file('SpeechToSignLanguagePage.kv')
    Builder.load_file('TextToSpeechPage.kv')
    MyApp().run()
