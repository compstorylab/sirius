import {Drawer} from "./drawer";

import {generateGraphChart} from "./graph_chart";
import {displayChart} from "./display_difference_chart";

(function(){

   let jsonUrl = <HTMLInputElement>document.getElementById('json-url'),
       body = document.getElementsByTagName("body")[0],
       uploadLink = document.getElementById("upload-link");

   let drawer:Drawer = Drawer.getInstance();

    uploadLink.addEventListener('click', () => {
       drawer.open();
    });

   // draw graph network
   if (jsonUrl && jsonUrl.value){
      // remove background image, replace with background color
      body.style.backgroundImage = 'None';
      // hide right panel by default when graph chart is generated
      drawer.close();

      generateGraphChart(jsonUrl);
      displayChart();
   }

}());