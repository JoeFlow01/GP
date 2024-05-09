from kivy.app import App
from kivy.uix.video import Video

class VideoApp(App):
    def build(self):
        video = Video(source='../Assets/test.mp4', state='play')
        return video

if __name__ == '__main__':
    VideoApp().run()
