from pyglet.window import key
from gym.envs.box2d.car_racing import CarRacing
import numpy as np

# Image Export
import os
from PIL import Image
from _aux import convert_action_to_nn

PATH = 'records/'
ACT_DESCRIPTIONS = [
        'Izquierda',
        'Centro',
        'Derecha',
        'Izq Gas',
        'Centro Gas',
        'Dcha Gas',
        'Freno'
]

def play(file_writer=None, pack_file=None, pack_name=None):
        
        a = np.array( [0.0, 0.0, 0.0] )

        def key_press(k, mod):
                global restart
                if k==0xff0d: restart = True
                if k==key.LEFT:  a[0] = -1.0
                if k==key.RIGHT: a[0] = +1.0
                if k==key.UP:    a[1] = +1.0
                if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        def key_release(k, mod):
                if k==key.LEFT  and a[0]==-1.0: a[0] = 0
                if k==key.RIGHT and a[0]==+1.0: a[0] = 0
                if k==key.UP:    a[1] = 0
                if k==key.DOWN:  a[2] = 0

        env = CarRacing()
        env.render()

        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release

        # Patrón que siguen los ficheros
        # pack_{num_episodes_in_pack}_{DDMMYYYYhhmmss of the last episode}.carrot

        # --- Preparamos el directorio de captura de imágenes ---
        if not os.path.exists(PATH):
                os.makedirs(PATH)

        index = 1
        while True:
                s = env.reset()
                total_reward = 0.0
                steps = 0
                restart = False
                while True:
                        a_nn = convert_action_to_nn(a)
                        print('{} - {}'.format(a_nn, ACT_DESCRIPTIONS[a_nn]))
                        next_s, r, done, info = env.step(a)

                        if file_writer:
                                np.savez('records/{}/{}.npz'.format(pack_name, index), s, next_s)
                                file_writer.writerow([a_nn, r, done])
                                pack_file.flush()

                        total_reward += r
                        if steps % 200 == 0 or done:
                                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                        steps += 1
                        index += 1
                        env.render()
                        s = next_s
                        if done or restart: break
        env.close()

if __name__ == "__main__":
    
        # Escritura de un pack
        import os
        import csv
        import datetime

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        today = datetime.datetime.today()

        today_str = today.strftime('%d%m%Y_%H%M%S')

        # --- Preparamos el directorio de captura de imágenes ---
        if not os.path.exists('records/pack_{}'.format(today_str)):
                os.makedirs('records/pack_{}'.format(today_str))

        # Patrón: pack_{ddMMYYYY_HHmmss}.carrots

        get_file_path = lambda x: os.path.join(BASE_DIR, 'records/pack_{}/{}.carrots'.format(x,x))

        with open(get_file_path(today_str), mode='w', encoding='utf-8') as pack:

                file_writer = csv.writer(pack, delimiter='|')
                file_writer.writerow(['Action', 'Reward', 'Done'])
                play(file_writer=file_writer, pack_file=pack, pack_name='pack_{}'.format(today_str))
        
        pack.close()