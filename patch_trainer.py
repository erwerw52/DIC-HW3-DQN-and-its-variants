import re

with open("d:/workspaces/aiot/dlr/HW3/src/trainer.py", "r", encoding="utf-8") as f:
    code = f.read()

# Add device support to _get_state
code = code.replace(
    'def _get_state(game):',
    'def _get_state(game, device="cpu"):\n    return torch.from_numpy(\n        game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0\n    ).float().to(device)'
)
code = code.replace('    return torch.from_numpy(\n        game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0\n    ).float()', '')

def patch_function(func_name, code):
    pattern = rf"def {func_name}\(.*?\):.*?yield i, loss\.item\(\), losses\[:\]"
    match = re.search(pattern, code, re.DOTALL)
    if not match:
        print(f"Skipping {func_name}")
        return code
    
    chunk = match.group(0)
    
    chunk = re.sub(r'def ([a-zA-Z0-9_]+)\((.*?\)):', r'def \1(\2, device="cpu"):', chunk)
    
    chunk = chunk.replace('losses    = []', 'losses    = []\n    rewards_list = []')
    chunk = chunk.replace('loss      = torch.tensor(0.0)', 'loss      = torch.tensor(0.0).to(device)')
    chunk = chunk.replace('status = 1', 'status = 1\n        episode_reward = 0.0')
    
    chunk = chunk.replace('reward = game.reward()', 'reward = game.reward()\n            episode_reward += reward')
    
    chunk = chunk.replace('losses.append(loss.item())', 'losses.append(loss.item() if hasattr(loss, "item") else loss)\n        rewards_list.append(episode_reward)')
    
    chunk = chunk.replace('yield i, loss.item(), losses[:]', 'yield i, (loss.item() if hasattr(loss, "item") else loss), losses[:], rewards_list[:]')
    chunk = chunk.replace('_get_state(game)', '_get_state(game, device)')
    
    chunk = chunk.replace('.to(device).to(device)', '.to(device)')
    if 'torch.tensor([' in chunk:   
        chunk = re.sub(r'torch\.tensor\(\[(.*?)\](, dtype=.*?)?\)', r'torch.tensor([\1]\2).to(device)', chunk)
    
    return code.replace(match.group(0), chunk)

for fn in ['train_naive_dqn', 'train_replay_dqn', 'train_double_dqn', 'train_dueling_dqn']:
    code = patch_function(fn, code)

with open("d:/workspaces/aiot/dlr/HW3/src/trainer.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Trainer patched")
