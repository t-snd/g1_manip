from g1_manip.cli import main

# add-images は描画を spawn の別プロセスで行い、子プロセスはこのモジュールを __mp_main__ として読み込むので、main は呼ばない
if __name__ == "__main__":
    main()
