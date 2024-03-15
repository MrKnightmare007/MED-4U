from flask import Flask, jsonify, render_template, request
from web3 import Web3
import time, json, os
from web3.middleware import geth_poa_middleware
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

ADMIN_ADDRESS = os.getenv("ADMIN_ADDRESS")
W3_PROVIDER = os.getenv("W3_PROVIDER")
SECRET_KEY = os.getenv("SECRET_KEY")
gasPrice = os.getenv("gasPrice")
gasLimit = os.getenv("gasLimit")

# Dictionary to store payment information with timestamps
payment_data = json.loads(open('mediplus-lite/payment_info.json', 'r').read())
app.config['SECRET_KEY'] = SECRET_KEY
w3 = Web3(Web3.HTTPProvider(W3_PROVIDER))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

scheduler = BackgroundScheduler()

def save_payment_info_to_json(payment_data):
    with open('mediplus-lite/payment_info.json', 'r+') as file:
        file_data = json.load(file)
        file_data.update(payment_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)

def timestamp_to_datetime(timestamp):
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

app.jinja_env.globals.update(timestamp_to_datetime=timestamp_to_datetime)

def check_payments():
    for payment_address, data in payment_data.items():
        timestamp = data['timestamp']
        current_time = time.time()

        if current_time - timestamp <= 900:
            balance = w3.eth.get_balance(payment_address)

            if balance > 0:
                try:
                    print(f"Payment received for {w3.from_wei(balance, 'ether')}")
                    send_payment_info_to_admin(payment_address, payment_data[payment_address]['private_key'], balance)
                except:
                    pass

scheduler.add_job(check_payments, 'interval', seconds=10)
scheduler.start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_payment_address', methods=['POST'])
def generate_payment_address():
    try:
        payment_amount = request.get_json().get('amount')

        if payment_amount is None:
            return jsonify({'error': 'Missing amount parameter'}), 400

        if w3.is_connected():
            new_address = w3.eth.account.create()
            payment_address = new_address.address
            private_key = w3.to_hex(new_address.key)
            timestamp = time.time()
            payment_data[payment_address] = {"amount": payment_amount, "timestamp": timestamp + 900, "private_key": private_key}
            save_payment_info_to_json(payment_data)
            return jsonify({
                'payment_address': payment_address,
                'amount': payment_amount,
                'valid_until': int(timestamp + 900)
            })
        else:
            return jsonify({'error': 'Failed to connect to Ethereum node'})
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/check_payment/<payment_address>')
def check_payment(payment_address):
    if w3.is_connected():
        if payment_address in payment_data:
            timestamp = payment_data[payment_address]['timestamp']
            current_time = time.time()

            if current_time - timestamp <= 900:
                balance = w3.eth.get_balance(payment_address)
                if balance > 0:
                    return jsonify({'status': '1', 'balance': w3.from_wei(balance, 'ether')})
                else:
                    return jsonify({'status': '0'})
            else:
                return jsonify({'error': 'Payment expired'})
        else:
            return jsonify({'error': 'Invalid payment address'})
    else:
        return jsonify({'error': 'Failed to connect to Ethereum node'})

def send_payment_info_to_admin(payment_address, private_key, balance):
    amount = gasLimit * w3.to_wei(gasPrice, 'gwei')
    amount = balance - amount
    admin_transaction = {
        'from': payment_address,
        'to': ADMIN_ADDRESS,
        'value': amount,
        'gas': gasLimit,
        'gasPrice': w3.to_wei(gasPrice, 'gwei'),
        'nonce': w3.eth.get_transaction_count(payment_address),
        'chainId': 97
    }

    signed_transaction = w3.eth.account.sign_transaction(admin_transaction, private_key)
    transaction_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)

    print(f"Payment information sent to admin. Transaction Hash: {transaction_hash.hex()}")

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
