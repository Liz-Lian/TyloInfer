

from pathlib import PurePosixPath

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class SPAStaticFiles(StaticFiles):


  async def get_response(self, path: str, scope: Scope) -> Response:

    headers = Headers(scope=scope)

    normalized = path.replace('\\', '/').strip('/')
    reserved = {'api', 'v1', 'realtime', 'health', 'docs', 'redoc', 'assets', 'static'}
    is_navigation = (
      scope['method'] in {'GET', 'HEAD'}
      and 'text/html' in headers.get('accept', '').lower()
      and headers.get('sec-fetch-dest', '') in {'', 'document', 'iframe'}
      and normalized.split('/', 1)[0] not in reserved
      and not PurePosixPath(normalized).suffix
      and '\\' not in scope['path']
      and '\x00' not in scope['path']
      and '..' not in scope['path'].split('/')
    )

    try:
      response = await super().get_response(path, scope)
    except HTTPException as error:
      if error.status_code != 404 or not is_navigation:
        raise
    else:

      if response.status_code != 404 or not is_navigation:
        return response


    return await super().get_response('index.html', scope)
