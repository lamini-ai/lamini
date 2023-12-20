import datetime
import concurrent.futures as cf
import logging

logger = logging.getLogger(__name__)

class Runner:
    def __call__(self, *args, **kwargs):
        num_servers = 4
        num_jobs_per_server = 200
        inputs = args[0] if args else kwargs.get("inputs")
        futures = []
        outputs = []

        if type(inputs) != list:
            return self.call(*args, **kwargs)

        input_groups = self.split_into_groups(inputs, num_jobs_per_server)

        with cf.ThreadPoolExecutor(max_workers=num_servers) as executor:
            init_task_len = min(num_servers, len(input_groups))
            for i in range(0, init_task_len):
                print(f"Submitting input group {i+1} out of {len(input_groups)} for inference")
                future = self.submit_sub_group(executor, input_groups[i], *args, **kwargs)
                futures.append(future)
                
            num_completed_futures = 0
            i = init_task_len

            while num_completed_futures < len(input_groups):
                completed_futures, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED, timeout=1200)
                num_completed_futures += len(completed_futures)
                
                for completed_future in completed_futures:
                    for item in completed_future.result():
                        outputs.append(item)
                    futures.remove(completed_future)
                    if i == len(input_groups): continue
                    print(f"Submitting input group {i+1} out of {len(input_groups)} for inference")
                    future = self.submit_sub_group(executor, input_groups[i], *args, **kwargs)
                    futures.append(future)
                    i += 1

        return outputs

    def submit_sub_group(self, executor, sub_inputs, *args, **kwargs):
        if args:
            args = (sub_inputs,) + args[1:] 
        else:
            kwargs['inputs'] = sub_inputs                    
        return executor.submit(lambda *a, **k: self.call(*a, **k), *args, **kwargs)

    def split_into_groups(self, inputs, group_size):
        groups = []
        for i in range(0, len(inputs), group_size):
            groups.append(inputs[i : i + group_size])
        return groups
