


export class Drawer {

    rightBar:HTMLElement;
    closeBtn:HTMLElement;
    fileSelectComponent:HTMLElement;
    imageTitleElement:HTMLElement;

    private static singleton:Drawer = null;

    public static getInstance():Drawer {
        if(Drawer.singleton == null) {
            Drawer.singleton = new Drawer();
        }
        return Drawer.singleton;
    }

    constructor(){
        this.rightBar = document.getElementById("right-bar");
        this.closeBtn = document.getElementById("close-btn");
        this.fileSelectComponent = document.getElementById("upload-box");
        this.imageTitleElement = document.getElementById("image-title");

        this.closeBtn.addEventListener('click', this.onCloseClick);
        this.fileSelectComponent.addEventListener('change', this.onFileSelectChange);
    }

    open(options:any=null) {
        if(!options) {
            this.openForUpload();
        }

        switch (options.mode){
            case 'upload':
                this.openForUpload();
                break;
            case 'plot':
                this.openForGraph(options);
                break;
        }
    }

    private openForUpload() {
        let rightContentBar = <HTMLElement>document.querySelector(".right-bar-content");
        let rightImageBar = <HTMLElement>document.querySelector(".right-bar-image");

        this.rightBar.hidden = false;
        rightContentBar.hidden = false;
        rightImageBar.hidden = true;
    }

    private openForGraph(options:any) {
        let rightContentBar = <HTMLElement>document.querySelector(".right-bar-content");
        let rightImageBar = <HTMLElement>document.querySelector(".right-bar-image");

        this.rightBar.hidden = false; // display right bar
        rightContentBar.hidden = true;
        rightImageBar.hidden = false; // display image bar, but hide content bar

        if (options.title) {
            this.imageTitleElement.innerText = options.title;
        }
    }

    close() {
        this.rightBar.hidden = true;
    }

    visible():boolean {
        return this.rightBar.hidden;
    }

    onCloseClick = () => {
        this.close();
    };

    onFileSelectChange = (event) => {
        let fileInput:HTMLInputElement = event.target as HTMLInputElement;
        let filesList:FileList = fileInput.files;
        if(filesList.length > 0) {
        let formElement:HTMLFormElement = document.getElementById('upload-form') as HTMLFormElement;
            formElement.submit();
        }
    };
}