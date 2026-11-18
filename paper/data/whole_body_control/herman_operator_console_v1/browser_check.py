"""Drive the local preview UI via Chromium CDP; never issues robot commands."""
import asyncio,base64,hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
import websockets
ROOT=Path.cwd();OUT=ROOT/'logs/recovery/operator-console-v1'
chrome=Path('/home/cpaxton/.cache/ms-playwright/chromium-1193/chrome-linux/chrome')
async def check():
 pages=json.loads(urllib.request.urlopen('http://127.0.0.1:9229/json/list').read());page=next(p for p in pages if p['type']=='page')
 async with websockets.connect(page['webSocketDebuggerUrl'],max_size=20*1024*1024) as ws:
  serial=0
  async def call(method,params=None):
   nonlocal serial
   serial+=1;identifier=serial;await ws.send(json.dumps({'id':identifier,'method':method,'params':params or {}}))
   while True:
    response=json.loads(await ws.recv())
    if response.get('id')!=identifier:continue
    if 'error' in response:raise RuntimeError(response['error'])
    return response.get('result',{})
  async def evaluate(expression):
   result=await call('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True})
   if 'exceptionDetails' in result:raise RuntimeError(result['exceptionDetails'])
   return result.get('result',{}).get('value')
  async def wait_for(expression):
   for _ in range(120):
    if await evaluate(expression):return
    await asyncio.sleep(.1)
   raise RuntimeError('browser condition timed out: '+expression)
  await call('Page.enable');await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
  await wait_for("typeof state !== 'undefined' && state?.connected && state.samples.hand?.fresh && document.getElementById('headImage').naturalWidth>0")
  before=await evaluate("({base:state.targets.base.pose.slice(),hand:state.targets.hands[0].pose.position.slice(),measured:state.samples.odom.position.slice(),commands:state.commands_sent})")
  await evaluate("document.querySelector('[data-mode=\"2\"]').click()")
  await wait_for('state.mode===2&&!busy')
  await call('Input.dispatchKeyEvent',{'type':'keyDown','key':' ','code':'Space','windowsVirtualKeyCode':32})
  await call('Input.dispatchKeyEvent',{'type':'keyDown','key':'w','code':'KeyW','windowsVirtualKeyCode':87})
  await asyncio.sleep(.38)
  await call('Input.dispatchKeyEvent',{'type':'keyUp','key':'w','code':'KeyW','windowsVirtualKeyCode':87})
  await call('Input.dispatchKeyEvent',{'type':'keyUp','key':' ','code':'Space','windowsVirtualKeyCode':32})
  await wait_for('!busy')
  hand_after=await evaluate('state.targets.hands[0].pose.position.slice()')
  if not hand_after[0]>before['hand'][0]:raise RuntimeError('hand keyboard did not edit target')
  orientation_before=await evaluate('state.targets.hands[0].pose.quaternion_xyzw.slice()')
  rect=await evaluate("(()=>{const r=document.querySelector('[data-action=\"rollPlus\"]').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()")
  await call('Emulation.setTouchEmulationEnabled',{'enabled':True,'maxTouchPoints':1})
  await call('Input.dispatchTouchEvent',{'type':'touchStart','touchPoints':[rect]})
  await asyncio.sleep(.25)
  await call('Input.dispatchTouchEvent',{'type':'touchEnd','touchPoints':[]})
  await call('Emulation.setTouchEmulationEnabled',{'enabled':False})
  await wait_for('!busy')
  if await evaluate('state.targets.hands[0].pose.quaternion_xyzw.slice()')==orientation_before:raise RuntimeError('orientation pointer control failed')
  await evaluate("document.querySelector('[data-mode=\"1\"]').click()")
  await wait_for('state.mode===1&&!busy')
  await call('Input.dispatchKeyEvent',{'type':'keyDown','key':' ','code':'Space','windowsVirtualKeyCode':32})
  await call('Input.dispatchKeyEvent',{'type':'keyDown','key':'a','code':'KeyA','windowsVirtualKeyCode':65})
  await asyncio.sleep(.35)
  await evaluate("window.dispatchEvent(new Event('blur'))")
  await wait_for('!busy')
  base_after=await evaluate('state.targets.base.pose.slice()')
  if not base_after[2]>before['base'][2]:raise RuntimeError('base turning keyboard did not edit target')
  frozen=await evaluate('state.revision');await asyncio.sleep(.3)
  if await evaluate('state.revision')!=frozen:raise RuntimeError('input continued after focus loss')
  await evaluate("document.querySelector('[data-mode=\"3\"]').click()")
  await wait_for('state.mode===3&&!busy')
  if await evaluate('state.targets.base.mask')!=[False]*3:raise RuntimeError('free-base mask not applied')
  await evaluate("document.getElementById('resetPreview').click()")
  await wait_for('!busy')
  reset=await evaluate('({targets:state.targets,origin:state.origin,measured:state.samples.odom.position.slice()})')
  if reset['targets']['base']['pose'][:2]!=reset['measured'][:2]:raise RuntimeError('preview reset did not copy measured base')
  await evaluate("document.querySelector('[data-mode=\"1\"]').click()")
  await wait_for('state.mode===1&&!busy')
  await evaluate("document.getElementById('headSelect').value='head_right'")
  await asyncio.sleep(.7)
  if not await evaluate("document.getElementById('headImage').naturalWidth>0"):raise RuntimeError('head right unavailable')
  overview=await evaluate("({title:document.title,connected:state.connected,motion_enabled:state.motion_enabled,commands_sent:state.commands_sent,base_mask:state.targets.base.mask,held_keys:keys.size,pointer_action:pointerAction,head_loaded:document.getElementById('headImage').naturalWidth,wrist_available:!!state.cameras.wrist?.fresh,stop_disabled:document.querySelector('.stop').disabled,horizontal_overflow:document.documentElement.scrollWidth>innerWidth,measured_base:state.samples.odom.position.slice()})")
  if overview['motion_enabled'] or overview['commands_sent'] or not overview['stop_disabled'] or overview['horizontal_overflow']:raise RuntimeError('UI authority/layout check failed')
  screenshot=await call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True});(OUT/'live-console.png').write_bytes(base64.b64decode(screenshot['data']))
  await call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
  await asyncio.sleep(.2)
  if await evaluate('document.documentElement.scrollWidth>innerWidth'):raise RuntimeError('mobile horizontal overflow')
  mobile=await call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True});(OUT/'mobile-console.png').write_bytes(base64.b64decode(mobile['data']))
  report={'format':'herman-console-browser-v1','live_robot_subscriptions_only':True,'hardware_motion_commands_sent':0,'checks':['live head stereo and telemetry','hand keyboard preview','base turn keyboard preview','focus loss clears held input','base-free mask','no motion/stop endpoint','no desktop/mobile horizontal overflow','touch orientation preview','reset preview to measured without motor command'],'before':before,'hand_after':hand_after,'base_after':base_after,'overview':overview,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/emet/visualization/operator_console.py',ROOT/'src/emet/visualization/operator_console.html',ROOT/'src/emet/robots/innate_mars/passive_relay.py',ROOT/'src/emet/robots/innate_mars/passive_ssh.py']}}
  (OUT/'browser-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
def main():
 with (OUT/'chromium.log').open('w') as log:
  process=subprocess.Popen(['taskset','-c','15',str(chrome),'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--disable-extensions','--remote-debugging-port=9229','--user-data-dir='+str(OUT/'chrome-profile'),'http://localhost:8770'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  try:
   for _ in range(60):
    try:urllib.request.urlopen('http://127.0.0.1:9229/json/list',timeout=1);break
    except OSError:time.sleep(.1)
   asyncio.run(check())
  finally:
   process.terminate()
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
if __name__=='__main__':main()
