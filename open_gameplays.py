if __name__ == "__main__":
    
        # Escritura de un pack
        import os
        import csv
        import datetime
        import numpy as np
        from PIL import Image

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        today = datetime.datetime.today()

        today_str = today.strftime('%d%m%Y%H%M%S')
        name = '29012019_010418'

        # Patrón: pack_{ddMMYYYYHHmmss}.carrots

        get_file_path = lambda x: os.path.join(BASE_DIR, 'records/pack_{}/{}.carrots'.format(x,x))
        get_nz_path = lambda x,y: os.path.join(BASE_DIR, 'records/pack_{}/{}.npz'.format(x,y))

        with open(get_file_path('29012019_010418'), mode='r', encoding='utf-8') as pack:

                file_reader = csv.reader(pack, delimiter='|')
                first = True
                counter = 0
                for l in file_reader:
                    
                    if counter == 0:
                        counter +=1
                        continue

                    npfile = np.load(get_nz_path(name, counter))

                    #Empieza en 3
                    if counter == 2:
                        s = npfile['arr_0']
                        i = np.asarray(s)
                        img = Image.fromarray(i, 'RGB')
                        img.show()

                        next_s = npfile['arr_1']
                        i = np.asarray(next_s)
                        img = Image.fromarray(i, 'RGB')
                        img.show()

                    counter += 1
                    
        
        pack.close()