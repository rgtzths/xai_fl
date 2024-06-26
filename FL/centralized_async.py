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
    patience_buffer = [-1]*patience

    if rank == 0:
        print("Running centralized async")
        print(f"Dataset: {dataset}")
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

    output = f"{dataset}/fl/centralized_async/{n_workers}_{epochs}_{batch_size}"
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
        
        total_n_batches = sum(node_weights)
        total_batches = epochs * total_n_batches

        node_weights = [weight/total_n_batches for weight in node_weights]
        results["times"]["sync"].append(time.time() - start)
        model_buff = bytearray(pickle.dumps(model.get_weights()))

    else:
        results = {"times" : {"train" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "epochs" : []}}

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))

        compr_data = pickle.dumps(len(train_dataset))

        comm.Send(compr_data, dest=0, tag=1000)

        total_batches = epochs * len(train_dataset)

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
    if rank == 0:
        exited_workers = 0
        latest_tag = 0
        for batch in range(total_batches):

            com_time = time.time()
            comm.Recv(grads_buff, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            results["times"]["comm_recv"].append(time.time() - com_time)

            load_time = time.time()
            grads = pickle.loads(grads_buff)
            results["times"]["conv_recv"].append(time.time() - load_time)

            source = status.Get_source()
            tag = status.Get_tag()

            if latest_tag < tag+1:
                latest_tag = tag+1

            behind_penalty = (tag+1 / latest_tag) #The more behind it is the less impact it will have, verry small penalization

            grads = [grad*node_weights[source-1]*behind_penalty for grad in grads] 

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            load_time = time.time()
            weights = pickle.dumps(model.get_weights())
            results["times"]["conv_send"].append(time.time() - load_time)

            comm_time = time.time()
            comm.Send(weights, dest=source, tag=tag)
            results["times"]["comm_send"].append(time.time() - comm_time)

            stop_buff = bytearray(pickle.dumps(stop))
            comm.Send(stop_buff, dest=source, tag=tag)

            if stop:
                exited_workers +=1
            if exited_workers == n_workers:
                break

            if (batch+1) % total_n_batches == 0 and not stop:

                results["times"]["epochs"].append(time.time() - epoch_start)

                print(f"\n End of batch {(batch+1)//n_workers} -> epoch {(batch+1)//total_n_batches}")

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

                if val_mcc >= early_stop or p_stop:
                    stop = True

                epoch_start = time.time()
            
    else:
        for batch in range(total_batches):

            train_time = time.time()

            with tf.GradientTape() as tape:
                x_batch_train, y_batch_train = train_dataset[batch % len(train_dataset)]
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
            comm.Recv(model_buff, source=0, tag=batch)
            results["times"]["comm_recv"].append(time.time() - com_time)

            load_time = time.time()
            weights = pickle.loads(model_buff)
            results["times"]["conv_recv"].append(time.time() - load_time)

            comm.Recv(stop_buff, source=0, tag=batch)
            stop = pickle.loads(stop_buff)

            model.set_weights(weights)

            if (batch+1) % len(train_dataset) == 0:
                results["times"]["epochs"].append(time.time() - epoch_start)
                epoch_start = time.time()

            if stop:
                break

    history = json.dumps(results)
    if rank==0:
        model.save(output/'trained_model.keras')
        with open(output/"server.json", "w") as f:
            f.write(history)

    else:
        with open(output/f"worker{rank}.json", "w") as f:
            f.write(history)

    
