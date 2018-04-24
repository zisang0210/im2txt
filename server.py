from sqlite3 import dbapi2 as sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash,jsonify
from werkzeug import secure_filename
import json,os
from datetime import *
import time  
import random  
import tensorflow as tf
from inference import *
from utils import vocabulary
from utils import att_nic_vocab

FLAGS = tf.app.flags.FLAGS


tf.flags.DEFINE_string('mode', 'att-nic',
                       'Can be att-nic or ours')
tf.flags.DEFINE_string('vocab_path', '../output/vocabulary.csv',
                       'Vocabulary file, be ../data/flickr8k/word_counts.txt for mode=ours')
tf.flags.DEFINE_string("faster_rcnn_path", "../data/frozen_faster_rcnn.pb",
                        "Faster r-cnn frozen graph")
tf.flags.DEFINE_string("region_lstm_path", "../data/frozen_lstm.pb",
                        "region attention based lstm forzen graph")
tf.flags.DEFINE_string("att_nic_path", "../data/frozen_att_nic.pb",
                        "region attention based lstm forzen graph")

if FLAGS.mode == 'ours':  
  faster_rcnn = FasterRcnnEncoder(FLAGS.faster_rcnn_path) 
  # build vocabulary file
  vocab = vocabulary.Vocabulary(FLAGS.vocab_path)
  lstm = LSTMDecoder(FLAGS.region_lstm_path,vocab,max_caption_length=20)
else:       
  vocab = att_nic_vocab.Vocabulary(5000, FLAGS.vocab_path)
  att_nic = ATT_NIC(FLAGS.att_nic_path,vocab,max_caption_length=20)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)


# database related operation
# Load default config and override config from an environment variable
app.config.update(dict(
    DATABASE='flaskr.db',
    DEBUG=True,
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='0000'
))
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

def init_db():
  """Creates the database tables."""
  with app.app_context():
    db = get_db()
    with app.open_resource('schema.sql', mode='r') as f:
      db.cursor().executescript(f.read())
    db.commit()

def get_db():
  """Opens a new database connection if there is none yet for the
  current application context.
  """
  if not hasattr(g, 'sqlite_db'):
    g.sqlite_db = connect_db()
  return g.sqlite_db

def connect_db():
  """Connects to the specific database."""
  rv = sqlite3.connect(app.config['DATABASE'])
  rv.row_factory = sqlite3.Row
  return rv

def add_entry(caption, raw_path, att_path):
  db = get_db()
  db.execute('insert into records (caption, raw_path, att_path) values (?, ?,?)',
               [caption, raw_path, att_path])
  db.commit()
  cur = db.execute('select last_insert_rowid() id')
  entries = cur.fetchall()
  return entries[0]['id']

def random_filename():
  """自动生成随机文件名"""
  nowTime = datetime.now().strftime("%Y%m%d%H%M%S")#生成当前的时间  
  randomNum = random.randint(0,100)#生成随机数n,其中0<=n<=100  
  if randomNum<=10:  
      randomNum = str(0) + str(randomNum)  
  uniqueNum = str(nowTime) + str(randomNum)  
  return uniqueNum

@app.teardown_appcontext
def close_db(error):
  """Closes the database again at the end of the request."""
  if hasattr(g, 'sqlite_db'):
      g.sqlite_db.close()


# im2txt
def generate_caption(data):
  """
  return 
  raw: raw picture in jpeg formate
  caption: generated image caption in string formate 
  attention: attention picture in jpeg formate

  """
  # results = classify(data)
  # return jsonify({'res': classify(data)})
  return data,'helloworld',data

# routes for user
@app.route('/')
def index():
  # Reading example picture information
  with open('static/images/examples/examples.txt', 'r') as f:
    imgs = json.load(f)
    
  return render_template('index.html', imgs=imgs)

@app.route('/upload/', methods=['GET', 'POST'])
def upload():
  file = request.files['file']
  if file and allowed_file(file.filename):
    # caption='hello world hello world hello world hello.'
    # raw,caption, attention = generate_caption(file.read())

    # get saved folder
    folder_name=datetime.now().strftime("%Y%m")
    folder_name='static/uploads/' + folder_name + '/'
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)
    # get saved filename
    filename = random_filename()
    raw_path = folder_name+filename+'0.jpeg'
    att_path = folder_name+filename+'1.jpeg'
    # save images
    file.save(raw_path)
    image_np = load_image_into_numpy_array(raw_path)
    if image_np is not None:
      if FLAGS.mode =='ours':
        box, feat = faster_rcnn.encode(image_np)
        caption, attention = lstm.decode(feat)
        lstm.show_attention(caption, attention,box, image_np, att_path)
      else:
        caption, attention = att_nic.decode(raw_path)
        att_nic.show_attention(caption, attention, image_np, att_path)
      # add record to sqlite
      id = add_entry(caption['caption'], '/'+raw_path, '/'+att_path)

      return jsonify({
        "code": 0,
        "msg": "succeed",
        "url": url_for('result', id=id)
        })   
    else:
      return jsonify({
        "code": 0,
        "msg": "fail",
        "url": url_for('index')
        })     

@app.route('/result/<id>')
def result(id):
  db = get_db()
  cur = db.execute('select id, caption, raw_path from records where id = '+id)
  entries = cur.fetchall()
  return render_template('result.html', r=entries[0])
 
@app.route('/rank/', methods=['GET', 'POST'])
def rank():
  db = get_db()
  db.execute("update records set rank = ?, updated_at = datetime(CURRENT_TIMESTAMP,'localtime') where id = ?",
               [request.args.get('rank'), request.args.get('id')])
  db.commit()
  return jsonify({
      "msg": "succeed",
      })     

# routes for admin
@app.route('/login/', methods=['GET', 'POST'])
def login():
  error = None
  if request.method == 'POST':
    if request.form['username'] != app.config['USERNAME']:
      error = 'Invalid username'
    elif request.form['password'] != app.config['PASSWORD']:
      error = 'Invalid password'
    else:
      session['logged_in'] = True
      return redirect(url_for('admin'))
  return render_template('login.html', error=error)

@app.route('/logout')
def logout():
  session.pop('logged_in', None)
  return redirect(url_for('login'))

# home page
@app.route('/admin/')
def admin():
  if not session.get('logged_in'):
    return redirect(url_for('login'))

  # return first page by default
  db = get_db()
  if request.args.get('rank'):
    cur = db.execute('select id, caption, raw_path, att_path, rank, created_at, \
      updated_at from records where deleted_at is null and rank = ? order by created_at desc',[request.args.get('rank')])
  else:
    cur = db.execute('select id, caption, raw_path, att_path, rank, created_at, \
      updated_at from records where deleted_at is null order by created_at desc')
  
  records = [dict((cur.description[idx][0], value)
               for idx, value in enumerate(row)) for row in cur.fetchall()]
  return render_template('admin.html', records=json.dumps(records),rank=request.args.get('rank',0))

if __name__ == '__main__':
  init_db()
  app.run(debug=False)