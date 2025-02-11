import torch
class blipTools():
    def activate_blip(self):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        """
        localにモデルを落とす方法
        rm -rf ~/.cache/huggingface
        python3 (local)
        model_name = "Salesforce/blip2-flan-t5-xl"
        save_dir = "/catkin_ws/src/master_thesis_modules/models/blip2-flan-t5-xl"
        # Processor の保存
        processor = Blip2Processor.from_pretrained(model_name)
        processor.save_pretrained(save_dir, push_to_hub=False)

        # モデルの保存
        model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        model.save_pretrained(save_dir, push_to_hub=False)
        """
        try:
            model_name="/catkin_ws/src/master_thesis_modules/models/blip2-flan-t5-xl"
            blip_processor = Blip2Processor.from_pretrained(model_name)#,revision="51572668da0eb669e01a189dc22abe6088589a24")
            print("load local model")
        except Exception:
            model_name="Salesforce/blip2-flan-t5-xl"
            blip_processor = Blip2Processor.from_pretrained(model_name)#,revision="51572668da0eb669e01a189dc22abe6088589a24")
            print("load online model")
        # model_name="Salesforce/blip2-opt-2.7b"
        blip_model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32) 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model.to(device)
        return blip_processor,blip_model,device
    

    def get_caption(self,blip_processor,blip_model,device,image,):
        caption_inputs = blip_processor(image, return_tensors="pt").to(device)

        # キャプションを生成する
        output = blip_model.generate(**caption_inputs)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        return caption
    
    def get_vqa(self,blip_processor,blip_model,device,image,question,confidence=False):
        inputs_what = blip_processor(image, text=question, return_tensors="pt").to(device, torch.float16)
        if confidence:
            outputs = blip_model.generate(**inputs_what, output_scores=True, return_dict_in_generate=True)
            # 生成された答え
            answer = blip_processor.decode(outputs.sequences[0], skip_special_tokens=True)
            scores = outputs.scores  # 各トークンごとのスコア
            probs = torch.nn.functional.softmax(scores[-1], dim=-1)  # 最後のトークンの確率分布
            confidence = probs.max().item()  # 最大確率（自信度）
            return answer,confidence
        else:

            generated_ids = blip_model.generate(**inputs_what, max_new_tokens=100)
            generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return generated_text

    def main(self):
        from PIL import Image
        blip_processor,blip_model,device=self.activate_blip()
        # image_path="/media/hayashide/MobileSensing/StopStandingObaachan/jpg/ytnpc2021h/StopStandingObaachan_ytnpc2021h_1724752870.0184581_003.jpg" # wheelchair
        image_path="/media/hayashide/MobileSensing/common/sample_images/infusion.jpg"
        # image_path="/media/hayashide/MobileSensing/Backup20241127/jpg/ytnpc2021h/Backup20241127_ytnpc2021h_1725059672.8465743_006.jpg"
        image=Image.open(image_path)
        caption=self.get_caption(blip_processor=blip_processor,blip_model=blip_model,device=device,image=image)
        print(caption)
        question="Question: How many people are there? Answer:"
        answer=self.get_vqa(blip_processor=blip_processor,blip_model=blip_model,device=device,image=image,question=question)
        print(answer)
        pass

    def segment_person(self,image_path, model):
        from PIL import Image
        import cv2
        import numpy as np
        """
        YOLOセグメンテーションを使って画像から人物の輪郭を抽出する。
        :param image_path: 入力画像のパス
        :param model: YOLOモデル
        :return: 人物部分だけを抽出したPIL.Imageオブジェクト
        """
        # 画像を読み込む
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # YOLOで推論
        results = model(image_path)
        
        # マスクを保持する配列を初期化
        final_mask = np.zeros(image_np.shape[:2], dtype=bool)

        # 元画像のサイズ
        original_height, original_width = image_np.shape[:2]

        # 人物（クラスID: 0）のセグメンテーションを適用
        for result in results:
            for mask, class_id in zip(result.masks.data, result.boxes.cls):
                if int(class_id) == 0:  # COCOの"person"クラスIDは0
                    # マスクをリサイズして元画像サイズに合わせる
                    resized_mask = cv2.resize(
                        mask.cpu().numpy().astype(np.uint8).T[::-1,:], 
                        (original_width, original_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    # リサイズしたマスクを論理和で結合
                    final_mask = final_mask | (resized_mask > 0)
        
        # マスクを元画像に適用して人物部分を抽出
        masked_image_np = image_np.copy()
        masked_image_np[~final_mask] = [255,255,255]  # 背景を黒で塗りつぶし

        # 抽出した画像をPillowに戻す
        masked_image = Image.fromarray(masked_image_np)
        masked_image=masked_image.rotate(-90)
        return masked_image
    
    def main_multiple_images(self):
        import os
        from glob import glob
        from PIL import Image
        import matplotlib.pyplot as plt
        image_paths=sorted(glob("/media/hayashide/MobileSensing/20241224_cliptest/*"))
        blip_processor,blip_model,device=self.activate_blip()
        image_paths=[k.replace("JPG","jpg") for k in image_paths]
        self.questions=[
            "Question: Is this person standing, vomitting or something else? Answer:",
            # "Question: Is this person extending their arms? Answer:",
            # "Question: Is this person touching their face? Answer:",
            # # "Question: Is this person extending their arms or keeping them bent? Answer:",
            # "Question: Is this person extending their legs? Answer:",
        ]
        from ultralytics import YOLO
        model=YOLO("yolo11x-seg.pt")
        for image_path in image_paths:
            print(f"# {os.path.basename(image_path)}")
            img=Image.open(image_path)
            img=self.segment_person(image_path,model)
            for q in self.questions:
                answer,confidence=self.get_vqa(blip_processor=blip_processor,blip_model=blip_model,device=device,image=img,question=q,confidence=True)
                print(f"q: {q}  answer: {answer}  conf:{confidence}")
            plt.imshow(img)
            plt.show()
            # plt.pause(1)
            # plt.close()
if __name__=="__main__":
    cls=blipTools()
    # cls.main()
    # cls.main_multiple_images()