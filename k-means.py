import tensorflow as tf
from data import make_dataset
import time
import matplotlib.pyplot as plt
'''
Implementation of k-means ++
'''

tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

NUM_CLUSTER = 9
NUM_ITERS = 30
NEW_CENTRIOD_THRESHOLD = 0.98

ds_1 = make_dataset(BATCH_SIZE=1, file_name='train_tf_record', split=False)
ds_2 = make_dataset(BATCH_SIZE=1, file_name='test_tf_record', split=False)
ds = ds_1.concatenate(ds_2)
NUM_SAMPLES = len(list(ds))
frame = []

for sample in ds:
    for num, x_min in enumerate(sample[1]):
        y_min = sample[2][num]
        x_max = sample[3][num]
        y_max = sample[4][num]
        width = tf.cast((x_max - x_min), tf.float32).numpy()
        height = tf.cast((y_max - y_min), tf.float32).numpy()
        frame.append((width, height))

# Calculate initial centroids (K-Means++)
start_index = tf.random.uniform(shape=(1,), minval=0, maxval=len(frame), dtype=tf.int32)
centriods = tf.gather(frame, indices=start_index).numpy().tolist()
print(centriods)
centriods_dist = tf.Variable(tf.zeros(shape=(NUM_SAMPLES, NUM_CLUSTER), dtype=tf.dtypes.float32), dtype=tf.float32)
for i in range(NUM_CLUSTER-1):
    for num_j, j in enumerate(frame):
        intersect = tf.minimum(j[0], centriods[i][0]) * tf.minimum(centriods[i][1], j[1])
        union = tf.reduce_prod(centriods[i]) + tf.reduce_prod(j) - intersect
        dist = tf.expand_dims((1 - (intersect / union)), axis=0)
        centriods_dist.scatter_nd_update([[num_j, i]],dist)

    sort = tf.sort(centriods_dist[:,i])

    new_centriod_dist = sort[int(NUM_SAMPLES * NEW_CENTRIOD_THRESHOLD)+ tf.random.uniform(shape=(), minval=0, maxval=5, dtype=tf.int64).numpy()]
    new_centriod_id = tf.where(tf.equal(centriods_dist[:,i], new_centriod_dist))
    if tf.reduce_prod(new_centriod_id.shape) > 1:
        new_centriod_id = new_centriod_id[0]
        print('multiple')
    if new_centriod_id > (NUM_SAMPLES - 1):
        new_centriod_id = NUM_SAMPLES - 1
        print('over')
    centriods.append(frame[tf.squeeze(new_centriod_id).numpy()])
    print(centriods)


plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x,y, c='r', marker='o')
sc1 = ax.scatter(x,y,marker='+', alpha=0.4)
plt.xlim(60,160)
plt.ylim(60,160)
plt.draw()
sc1.set_offsets(frame)
sc.set_offsets(centriods)
fig.canvas.draw_idle()
plt.pause(0.1)

dist_frame = tf.Variable(tf.zeros(shape=(NUM_SAMPLES, NUM_CLUSTER), dtype=tf.dtypes.float32), dtype=tf.float32)

centriods_output = []
for iter_num in range(NUM_ITERS):
    # Calculate initial distance
    begin = time.time()
    for num_i, i in enumerate(frame):
        for num_j, j in enumerate(centriods):
            intersect = tf.minimum(i[0], j[0]) * tf.minimum(i[1], j[1])
            union = tf.reduce_prod(i) + tf.reduce_prod(j) - intersect
            dist = tf.expand_dims((1 - (intersect / union)), axis=0)
            dist_frame.scatter_nd_update([[num_i,num_j]],dist)
    # Calculate the new centroids.
    cluster = tf.argmin(dist_frame, axis=1)
    attributes = dict()
    all_samples_indices = list(range(NUM_SAMPLES))
    for i in range(NUM_CLUSTER):
        mask = tf.equal(cluster, i)
        attribute = tf.boolean_mask(all_samples_indices, mask)
        if len(attribute) == 0:
            continue
        else:
            sub_frame = tf.gather(frame, attribute)
            centriods[i] = tf.reduce_mean(sub_frame, axis=0).numpy()
            attributes['{}'.format(i)] = attribute
        sc.set_offsets(centriods[i])
    end = time.time()
    centriods_output = list(map(lambda x: (round(list(x)[0]), round(list(x)[1])), centriods))
    print('Iterations {}, time:{}'.format(iter_num+1, end-begin))
    print('New centriods are \n {}'.format(centriods_output))
    sc1.set_offsets(frame)
    sc.set_offsets(centriods)
    fig.canvas.draw_idle()
    plt.pause(0.1)

centriods_output.sort(key=lambda x: x[0]*x[1])
print('Final ordered centriods are \n {}'.format(centriods_output))
plt.waitforbuttonpress()



