#-*-coding:utf-8-*-
import sys,os
Basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Basedir)

def Calculate_H_M_S(seconds):
    minutes = seconds / 60
    hours = int(minutes / 60)
    minutes = int(minutes - hours * 60)
    seconds = int(seconds - minutes * 60 - hours * 60 * 60)
    return (str(hours),str(minutes),str(seconds))

def main():
    pass

if __name__ == '__main__':
    main()