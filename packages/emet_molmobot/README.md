# Emet MolmoBot wrapper

Optional bridge to upstream [MolmoBot](https://github.com/allenai/MolmoBot) policy serving.

Install MolmoBot in a dedicated venv, then:

```bash
emet molmobot serve-policy --hf-repo allenai/MolmoBot-DROID --action-type joint_pos
```

This delegates to upstream ``serve_molmo.py`` when ``MOLMOBOT_PYTHON`` or ``.venv-molmobot`` is configured.

See [docs/datasets/molmobot.md](../../docs/datasets/molmobot.md).
