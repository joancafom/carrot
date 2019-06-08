import serial, time

SERIAL_PORT = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_85735313233351D092A0-if00"
SERIAL_FRQ = 9600

# Note: Serial Communications must be performed in binary

try:
    # Try to connect to the Board through Serial port
    arduino = serial.Serial(SERIAL_PORT, SERIAL_FRQ)
    time.sleep(2)

    while True:

        print("Introduce un número del 0 al 9: \n")
        read_number = input()

        try:
            # Check if it's a number and is between the boundaries
            checker = int(read_number)
            if checker >= 0 and checker <= 9:
                arduino.write(read_number.encode())
                
                # Read board's response
                response = arduino.readline()
                print("Respuesta: " + str(response))
            else:
                print("El número no se encuentra entre 0 y 9, vuelve a intentarlo")

        except Exception as e:
            print("Se ha producido un error. Posiblemente no ha introducido un número")
            print(e)

    arduino.close()

except Exception as e:
    print("No se ha podido abrir una conexión con el puerto y la frecuencia indicadas:")
    print("\t - Puerto: \"{}\"".format(SERIAL_PORT))
    print("\t - Frecuencia: {} baudios".format(SERIAL_FRQ))
