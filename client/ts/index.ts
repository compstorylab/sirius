import * as d3 from "d3";
import {generateGraghChart} from "./graph_chart";
import {displayChart} from "./display_difference_chart";

(function(){

   let jsonUrl = <HTMLInputElement>document.getElementById('json-url'),
       body = document.getElementsByTagName("body")[0],
       closeBtn = document.getElementsByClassName("close-btn")[0],
       rightBar = document.getElementById("right-bar"),
       uploadLink = document.getElementById("upload-link"),
       fileSelectComponent = document.getElementById("upload_box");

   // display the right panel
   uploadLink.addEventListener("click", function(){
      let rightContentBar = <HTMLElement>document.querySelector(".right-bar-content"),
          rightImageBar = <HTMLElement>document.querySelector(".right-bar-image");
      rightBar.hidden = false;
      rightContentBar.hidden = false;
      rightImageBar.hidden = true;

   });
   // close the right panel
   closeBtn.addEventListener("click", function(evt){
      rightBar.hidden = true;

   });

   fileSelectComponent.addEventListener('change', function (event) {
      let fileInput:HTMLInputElement = event.target as HTMLInputElement;
      let filesList:FileList = fileInput.files;
      if(filesList.length > 0) {
         let formElement:HTMLFormElement = document.getElementById('upload_form') as HTMLFormElement;
         formElement.submit();
      }
   });

   // draw graph network
   if (jsonUrl && jsonUrl.value){
      // remove background image, replace with background color
      body.style.backgroundImage = 'None';
      // hide right panel by default when graph chart is generated
      if (rightBar){
         rightBar.hidden = true;
      }
      generateGraghChart(jsonUrl);
      displayChart();
   }

}());