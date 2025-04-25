from npcpy.npc_compiler import NPC
import os 
from sqlalchemy import create_engine
sibiji_path = os.path.expanduser("~/.npcsh/npc_team/sibiji.npc")
if not os.path.exists(sibiji_path):
    sibiji_path = __file__.getparent() / "npc_team/sibiji.npc"
db= create_engine("sqlite:///"+os.path.expanduser('~/npcsh_history.db'))

sibiji = NPC(file = sibiji_path, db_conn = db)


