from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle


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

    def go_to_SpeechToSignLanguagePage(self):
        print("Speech to SignLang")

    def go_to_SignLanguageToSpeechPage(self):
        print("Trying to go to SingLang to speech page")


class MyApp(App):
    def build(self):
        self.screen_manager = ScreenManager()
        self.screen_manager.add_widget(SplashScreen(name='SplashScreen'))
        self.screen_manager.add_widget(HomePage(name='HomePage'))
        Clock.schedule_once(self.go_to_home, 2)
        return self.screen_manager

    def go_to_home(self, dt):
        self.screen_manager.current = 'HomePage'



class SplashScreen(Screen):
    pass


class HomePage(Screen):
    pass


if __name__ == '__main__':
    Builder.load_file('HomePage.kv')
    MyApp().run()
