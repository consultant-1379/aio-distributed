from kubernetes import client, config, watch
import threading

def hello():
    print('hello')

def update_config():
    while True:
        try:
            config.load_kube_config()
            v1 = client.CoreV1Api()
            api_response = v1.list_namespaced_config_map(namespace='adtoolbox',
                                                         label_selector='config=threshold',
                                                         timeout_seconds=10)
            threshold_str = (api_response.items[0].data).get('threshold')
            resource_version = api_response.metadata.resource_version
            threshold_value = int(threshold_str)
            print('threshold_value ', threshold_value)
            print(api_response.metadata.resource_version)
            w = watch.Watch()
            for event in w.stream(func=v1.list_namespaced_config_map,
                                  namespace='adtoolbox',
                                  label_selector='config=threshold',
                                  resource_version=resource_version):
                if event["type"] == "MODIFIED":
                    print('config map modified')
                    threshold_str = event['object'].data.get('threshold')
                    threshold_value = int(threshold_str)
                    print('update thresholding value to ',threshold_value)
                if event["type"] == "DELETED":
                    print("config map is deleted")
                    w.stop()

        except Exception as e:
            print('exception ', e, '\n')
            pass


if __name__ == '__main__':
    thread = threading.Thread(target=update_config)
    thread.start()
    print('hello here ?')
