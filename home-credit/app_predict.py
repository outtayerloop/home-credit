import subprocess
import requests
from pyngrok import ngrok

if __name__ == '__main__':
    subprocess.run('mlflow models serve -m ./mlruns/0/b3544ea5fefe4efc8c94417fd58c0007/artifacts/model -p 1234', shell=True)
    #subprocess.run('mlflow models serve -m ./mlruns/0/b3544ea5fefe4efc8c94417fd58c0007/artifacts/model -p 1234 &')
    ngrok.kill()
    ngrok.set_auth_token('')
    ngrok_tunnel = ngrok.connect(addr='1234', proto='http', bind_tls=True)
    print('MLflow Models Serve URL : ', ngrok_tunnel.public_url)
    #system('mlflow models serve -m ./mlruns/0/b3544ea5fefe4efc8c94417fd58c0007/artifacts/model -p 1234')