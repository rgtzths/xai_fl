import json
import pathlib
import time
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

tf.keras.utils.set_random_seed(42)

def run(
    dataset_util, 
    optimizer,
    early_stop,
    learning_rate,
    batch_size,
    epochs,
    patience,
    min_delta
):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_workers = comm.Get_size()-1
    status = MPI.Status()
    pickle =  MPI.Pickle()
    stop = False
    stop_buff = bytearray(pickle.dumps(stop))

    dataset = dataset_util.name
    patience_buffer = [0]*patience


    if rank == 0:
        print("Running centralized sync")
        print(f"Dataset: {dataset}")
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

    output = f"{dataset}/fl/centralized_sync/{n_workers}_{epochs}_{batch_size}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)

    model = dataset_util.create_model()
    optimizer = optimizer(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    start = time.time()
    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "sync" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "global_times" : []}}
        node_weights = [0]*n_workers

        X_cv, y_cv = dataset_util.load_validation_data()
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv)).batch(batch_size)

        size_buff = bytearray(pickle.dumps(len(X_cv)))
        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights

        for node in range(n_workers):
            comm.Recv(size_buff, source=MPI.ANY_SOURCE, tag=1000, status=status)
            n_examples = pickle.loads(size_buff)
    
            node_weights[status.Get_source()-1] = n_examples
        
        sum_n_batches = sum(node_weights)
        total_n_batches = max(node_weights)
        total_batches = epochs * total_n_batches

        node_weights = [weight/sum_n_batches for weight in node_weights]
        results["times"]["sync"].append(time.time() - start)
        model_buff = bytearray(pickle.dumps(model.get_weights()))

    else:
        results = {"times" : {"train" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "epochs" : []}}

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))

        compr_data = pickle.dumps(len(train_dataset))

        comm.Send(compr_data, dest=0, tag=1000)

        total_n_batches = len(train_dataset)
        total_batches = epochs * total_n_batches

        model_buff = bytearray(pickle.dumps(model.get_weights()))

    comm.Bcast(model_buff, root=0)

    if rank != 0:
        weights = pickle.loads(model_buff)

        model.set_weights(weights)
    else:
        results["times"]["sync"].append(time.time() - start)

        #Get gradient size
        x_batch, y_batch = list(val_dataset.take(1))[0]

        with tf.GradientTape() as tape:

            logits = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, logits)

        grads_buff = bytearray(pickle.dumps(tape.gradient(loss_value, model.trainable_weights)))

    epoch_start = time.time()
    for batch in range(total_batches):
        weights = []
        if rank == 0:
            avg_grads = []
            for _ in range(n_workers):

                com_time = time.time()
                comm.Recv(grads_buff, source=MPI.ANY_SOURCE, tag=batch, status=status)
                results["times"]["comm_recv"].append(time.time() - com_time)

                load_time = time.time()
                grads = pickle.loads(grads_buff)
                results["times"]["conv_recv"].append(time.time() - load_time)

                source = status.Get_source()

                if not avg_grads:
                    avg_grads = [grad*node_weights[source-1] for grad in grads]
                else:
                    for idx, weight in enumerate(grads):
                        avg_grads[idx] += weight*node_weights[source-1]
            
            optimizer.apply_gradients(zip(avg_grads, model.trainable_weights))

            load_time = time.time()
            model_buff = bytearray(pickle.dumps(model.get_weights()))
            results["times"]["conv_send"].append(time.time()- load_time)
            
        else:
            train_time = time.time()

            x_batch_train, y_batch_train = train_dataset[batch % len(train_dataset)]

            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            results["times"]["train"].append(time.time() - train_time)

            load_time = time.time()
            grads = pickle.dumps(grads)
            results["times"]["conv_send"].append(time.time() - load_time)
            
            comm_time = time.time()
            comm.Send(grads, dest=0, tag=batch)
            results["times"]["comm_send"].append(time.time() - comm_time)

        com_time = time.time()
        comm.Bcast(model_buff, root=0)

        if rank == 0:
            results["times"]["comm_send"].append(time.time() - com_time)
        else:
            results["times"]["comm_recv"].append(time.time() - com_time)

            conv_time = time.time()
            weights = pickle.loads(model_buff)
            results["times"]["conv_recv"].append(time.time() - conv_time)

            model.set_weights(weights)
        
        if rank == 0:
                stop_buff = bytearray(pickle.dumps(stop))

        comm.Bcast(stop_buff, root=0)

        if rank != 0:
            stop = pickle.loads(stop_buff)

        if stop:
            break

        if (batch+1) % total_n_batches == 0:

            if rank == 0:
                results["times"]["epochs"].append(time.time() - epoch_start)

                print(f"\n End of batch {batch+1} -> epoch {(batch+1) // total_n_batches}")
                predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
                val_f1 = f1_score(y_cv, predictions, average="macro")
                val_mcc = matthews_corrcoef(y_cv, predictions)
                val_acc = accuracy_score(y_cv, predictions)

                results["acc"].append(val_acc)
                results["f1"].append(val_f1)
                results["mcc"].append(val_mcc)
                results["times"]["global_times"].append(time.time() - start)
                print("- val_f1: %6.3f - val_mcc %6.3f - val_acc %6.3f" %(val_f1, val_mcc, val_acc))
                patience_buffer = patience_buffer[1:]
                patience_buffer.append(val_mcc)

                p_stop = True
                for value in patience_buffer[1:]:
                    if abs(patience_buffer[0] - value) > min_delta:
                        p_stop = False 
    
                if (val_mcc > early_stop or p_stop) and (batch+1) // n_batches > 10:
                    stop = True

            else:
                results["times"]["epochs"].append(time.time() - epoch_start)

            epoch_start = time.time()


    history = json.dumps(results)
    if rank==0:
        model.save(output/'trained_model.keras')
        with open(output/"server.json", "w") as f:
            f.write(history)

    else:
        with open(output/f"worker{rank}.json", "w") as f:
            f.write(history)
