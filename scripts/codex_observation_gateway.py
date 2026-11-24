# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Local file-backed observation/decision bridge for Codex-supervised tool trials.

No model inference or robot commands occur in this bridge. An interactive
supervisor reads recorded images and writes a decision JSON for each request.
It deliberately records only the same text/images supplied to the VLM API.
"""
import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--response-timeout-s', type=float, default=85.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lock, counter = threading.Lock(), [0]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with lock:
                counter[0] += 1
                request_id = counter[0]
            output = args.output/f'{request_id:04d}'
            output.mkdir()
            (output/'request.json').write_text(json.dumps(payload)+'\n')
            texts, images = [], []
            for message in payload.get('messages', []):
                content = message.get('content', [])
                if isinstance(content, str):
                    texts.append(content)
                    continue
                for item in content:
                    if item['type'] == 'text':
                        texts.append(item['text'])
                    elif item['type'] == 'image_url':
                        data_url = item['image_url']['url']
                        if not data_url.startswith('data:image/png;base64,'):
                            raise ValueError('only inline PNG observations are supported')
                        path = output/f'image-{len(images)}.png'
                        path.write_bytes(base64.b64decode(data_url.split(',', 1)[1]))
                        images.append(str(path))
            (output/'observation.json').write_text(json.dumps({
                'id': request_id, 'texts': texts, 'images': images,
                'decision_path': str(output/'decision.json'), 'input_source': 'same VLM request text/images',
                'supervisor': 'interactive Codex; not an autonomous served model'}, indent=2)+'\n')
            print(json.dumps({'pending': request_id, 'observation': str(output/'observation.json')}), flush=True)
            deadline = time.monotonic()+args.response_timeout_s
            decision_path = output/'decision.json'
            while not decision_path.exists() and time.monotonic() < deadline:
                time.sleep(.1)
            if not decision_path.exists():
                decision = {'unavailable': 'interactive supervisor timeout'}
                (output/'timeout.json').write_text(json.dumps(decision)+'\n')
            else:
                decision = json.loads(decision_path.read_text())
            response = {'id': f'codex-observation-{request_id}', 'model': 'codex-interactive',
                'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(decision)}}]}
            encoded = json.dumps(response).encode()
            (output/'response.json').write_bytes(encoded)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(json.dumps({'listening': args.port, 'output': str(args.output)}), flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
