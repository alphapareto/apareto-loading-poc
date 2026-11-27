#!/usr/bin/env python3
"""Spread one client's requests over several apareto servers.

    python3 runners/lcb/balance.py --port 8100 --log <new file> http://127.0.0.1:8000 http://127.0.0.1:8001

A harness that takes one base URL (lighteval's litellm endpoint) talks to this address instead of a server. Each
POST goes to the server with the fewest requests in flight (the first listed on a tie), and the response is passed
back unchanged, streamed or not. GET requests (/v1/models, /health) are answered by the first server. Nothing in a
request or a response is modified. One line per finished POST is appended to --log: the server it went to, the
status and the seconds it took.
"""
import argparse
import asyncio
import json
import time

import aiohttp
from aiohttp import web

HOP_BY_HOP = {'connection', 'keep-alive', 'transfer-encoding', 'content-length', 'host', 'content-encoding'}


def forwardable(headers):
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


async def handle(request):
    app = request.app
    backends, inflight = app['backends'], app['inflight']
    index = 0 if request.method != 'POST' else min(range(len(backends)), key=lambda i: (inflight[i], i))
    body = await request.read()
    started = time.time()
    status = None
    if request.method == 'POST':
        inflight[index] += 1
    try:
        async with app['client'].request(request.method, backends[index] + request.rel_url.path_qs, data=body,
                                         headers=forwardable(request.headers)) as upstream:
            status = upstream.status
            response = web.StreamResponse(status=status, headers=forwardable(upstream.headers))
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response
    except (aiohttp.ClientError, asyncio.TimeoutError) as error:
        status = 502
        return web.json_response({'error': {'message': f'{backends[index]}: {error!r}', 'type': 'bad_gateway'}},
                                 status=502)
    finally:
        if request.method == 'POST':
            inflight[index] -= 1
            app['log'].write(json.dumps({'t': round(started, 3), 'path': request.path, 'server': index,
                                         'backend': backends[index], 'status': status,
                                         'seconds': round(time.time() - started, 3),
                                         'in_flight_after': list(inflight)}) + '\n')
            app['log'].flush()


async def start(app):
    app['client'] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), auto_decompress=False)


async def stop(app):
    await app['client'].close()
    app['log'].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--log', required=True, help='a new file; one JSON line per finished POST')
    parser.add_argument('backends', nargs='+', help='server base URLs without /v1, e.g. http://127.0.0.1:8000')
    args = parser.parse_args()
    app = web.Application(client_max_size=1 << 28)
    app['backends'] = [b.rstrip('/') for b in args.backends]
    app['inflight'] = [0] * len(args.backends)
    app['log'] = open(args.log, 'x')
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    app.router.add_route('*', '/{tail:.*}', handle)
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)


if __name__ == '__main__':
    main()
