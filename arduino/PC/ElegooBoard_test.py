from ElegooBoard import ElegooBoard
import time

if __name__ == "__main__":
    
    instructions = [-1, 0, 1, 2, 3, 4, 5, None, 'A', 5.5]
    i_to_str = ['Izquierda', 'Centro', 'Derecha', 'Centro-Gas', 'Freno']

    print("\n -- ELEGOOBOARD COMMUNICATION TESTER --\n")
    
    board = ElegooBoard()
    
    if board.open():
        print("Conexión establecida")
        print("Pulsa una tecla para iniciar el envío de instrucciones")
        _ = input()

        for i,e in enumerate(instructions):
            print("=== Probando instrucción {}/{}: {}".format((i+1), len(instructions), e))
            
            if(type(e) is int and e >= 0 and e < len(i_to_str)):
                print("({})".format(i_to_str[e]))
            
            try:
                board.send_directions(e)
            except Exception as e:
                print(e)

            time.sleep(3)
            
    print("\nCerrando la conexión")
    board.close()

    print("¿Se ha cerrado la conexión?: {}".format(board.isOpen()))