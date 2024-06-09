let chat = document.getElementById("chatcontent");
let chats = document.getElementById("chatarea");
let sidebar = document.getElementById("innersidebar");
let messageinput = document.getElementById("msg");
let sendbutton = document.getElementById("send");
let sendbuttonp = document.getElementById("sendp");
let nohistory = document.getElementById("No_history");
let starttext = document.getElementById("starting");
let indicator = document.getElementById("indicate");
let UploadFile = document.getElementById("uploadfile");
let spinner = document.getElementById("spinner");
let frontloading = document.getElementById("Loading");


async function sendFile()
  { let File = document.getElementById("file");
      if(File.files.length!==0){
      spinner.style.display="block";
        frontloading.style.display="flex";
        UploadFile.style.display = "none";
        chat.innerHTML = "";

        let formdata = new FormData();
        formdata.append('file',File.files[0])
        console.log(formdata);
    
        await fetch('/Upload_File/', {
      method: 'POST',
      body: formdata,
})
  .then(response => response.json())
  .then(data => {
    spinner.style.display = "none";
    frontloading.style.display="none";
    indicator.innerText=data.response;
    sendbutton.disabled = false;
    sendbuttonp.style.opacity = "1";
    console.log(data.response);
    }
  )

      }
      else
      alert("Please Select a File");
}




async function answer_to_query()
{
    if(messageinput.value!="")
   {
    let val = messageinput.value;
    starttext.style.display="none";
    chat.style.display="block";
    chat.innerHTML+=`<p>You: ${messageinput.value}</p>`;
    sendbutton.disabled = true;
    sendbuttonp.style.opacity = "0.3";
    messageinput.value="";
    chats.scrollTop = chats.scrollHeight;

  //Fetch method uses for api call
    fetch('/SolveQuery/',{
    method:'POST',
    headers:{
      'Content-Type':'application/x-www-form-urlencoded'
    },
    body:"query="+encodeURIComponent(val) //Sends the user query
    })
    .then(response => response.json())
    .then(data => {
    sendbutton.disabled = false;
    sendbuttonp.style.opacity = "1";

    if((data.answer.includes("An Error Occurred"))===false) //For Handling Error If this false the retrived data will be shown in interface
    {
    chat.innerHTML+=`Ai: ${data.answer}`;
    chats.scrollTop = chats.scrollHeight;
    if('chatdata' in data)
    {
      if(nohistory)
      sidebar.innerHTML="";

      sidebar.innerHTML+=`<button id="historychat" onclick='sendid(${data.chatdata.chatid})'>${data.chatdata.title}...</button>`;

      sidebar.scrollTop=-sidebar.scrollHeight;
    }
    }

    else
    {
      chat.innerHTML+=`<h3 style="color:red;">${data.answer}</h3>`;
      sendbutton.disabled = true;
      sendbuttonp.style.opacity = "0.3";
    }
    


})
.catch((error) => {

  console.error('Error:', error);
  
});
}

}

function sendid(chatid)
 {
    spinner.style.display = "block";
    frontloading.style.display="flex";
    UploadFile.style.display = "none";
    chat.innerHTML="";
    indicator.innerText = "";

    fetch(`/history/${chatid}`)
    .then(response=>response.json())
    .then(data=>{

      spinner.style.display = "none";
      frontloading.style.display="none";
      chat.style.display="block";
      sendbutton.disabled = false;
      sendbuttonp.style.opacity = "1";

      if (data.history!==false)
      {
        indicator.innerText=`File: ${data.filename} loaded successfully`;
        data.history.forEach(element => {
        
        if('user'===element.role)
        chat.innerHTML+=`<p>You: ${element.parts}</p>`;
        else
        chat.innerHTML+=`Ai: ${element.parts}`;
        chats.scrollTop = chats.scrollHeight;
  
      });
    }
    else
    {
        indicator.innerText="Failed to Load the File Try Again...";
    }
    })
  }







function showsidebar()
{
    let sidebar = document.getElementById("sidebar");
    sidebar.style.transform = "none";
}

function hidesidebar()
{
    let sidebar = document.getElementById("sidebar");
    sidebar.style.transform = "translateX(-380px)";
}

function newchat()
{
  window.location.href="/";
}

function newRAGchat()
{
  window.location.href="/RAG";
}