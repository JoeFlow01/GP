from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.uix.image import AsyncImage
from kivy.graphics import Color, Rectangle


class CameraPage(BoxLayout):
    def __init__(self, **kwargs):
        super(CameraPage, self).__init__(**kwargs)
        self.orientation = 'vertical'

        # Create a blue bar for the label
        with self.canvas.before:
            Color(0.2, 0.6, 1, 1)  # Blue color
            self.rect = Rectangle(size=self.size, pos=self.pos)

        # Bind the rectangle's size and position to the layout's size and position
        self.bind(size=self._update_rect, pos=self._update_rect)

        # Add a label in the blue bar
        self.label = Label(text="Camera Page", color=(1, 1, 1, 1), size_hint=(1, None), height=50)
        self.add_widget(self.label)

        # Create a camera widget
        self.camera = Camera(play=True)
        self.add_widget(self.camera)

        # Add an AsyncImage widget for the GIF at the bottom
        try:
            self.gif_image = AsyncImage(source='../Assets/audiox.gif')
            self.add_widget(self.gif_image)
        except Exception as e:
            print(f"Error loading GIF: {e}")

    def _update_rect(self, instance, value):
        self.rect.pos = self.pos
        self.rect.size = self.size


class CameraApp(App):
    def build(self):
        return CameraPage()


if __name__ == '__main__':
    CameraApp().run()
