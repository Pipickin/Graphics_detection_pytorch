from VideoComp import VideoComp

if __name__=='__main__':
    video_path = '/media/shchetkov/HDD/media/videos/task2/bask_test/bask_00_31_05_50.mp4'
    model_path = '../model/weights/512/encoder.pth'
    threshold = 40
    step = 2

    video_comp = VideoComp(video_path, model_path, threshold, step)
    video_comp.compare_cap()
    # frame1 = video_comp.get_frame(100)
    # frame2 = video_comp.get_frame(101)
    #
    # output1 = video_comp.apply_encoder(frame1)
    # output2 = video_comp.apply_encoder(frame2)
    # print(video_comp.compare_encoded_frames(output1, output2))
